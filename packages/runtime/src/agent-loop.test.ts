import { describe, expect, it } from 'vitest';
import { type AgentEvent, AgentLoop } from './agent-loop.ts';
import type { AgentTool, AgentToolResult } from './llm-types.ts';
import {
	type FauxResponseStep,
	fauxAssistantMessage,
	fauxProvider,
	fauxText,
	fauxToolCall,
} from './test-utils/faux.ts';

function textResult(text: string, terminate?: boolean): AgentToolResult {
	return {
		content: [{ type: 'text', text }],
		details: {},
		...(terminate ? { terminate } : {}),
	};
}

function tool(
	name: string,
	execute: AgentTool['execute'],
	extra: Partial<AgentTool> = {},
): AgentTool {
	return {
		name,
		label: name,
		description: `The ${name} tool.`,
		parameters: { type: 'object', properties: {} },
		execute,
		...extra,
	};
}

/** A loop on the faux model, with every event recorded as `type` or `type:role`. */
function fauxLoop(responses: FauxResponseStep[], tools: AgentTool[] = []) {
	const faux = fauxProvider();
	faux.setResponses(responses);
	const { createAdapter } = faux.provider;
	const model = faux.getModel();
	if (!createAdapter || !model) throw new Error('The faux provider has no adapter or model.');
	const loop = new AgentLoop({
		initialState: {
			systemPrompt: 'Be brief.',
			model,
			tools,
			thinkingLevel: 'off',
		},
		sessionId: 'conv-1',
		createAdapter: async (model) =>
			createAdapter(model, { auth: undefined, promptCacheKey: 'conv-1' }),
	});
	const events: string[] = [];
	const recorded: AgentEvent[] = [];
	loop.subscribe((event) => {
		recorded.push(event);
		const role = 'message' in event && event.type !== 'turn_end' ? `:${event.message.role}` : '';
		if (event.type !== 'message_update') events.push(`${event.type}${role}`);
	});
	return { loop, faux, events, recorded };
}

describe('AgentLoop', () => {
	it("runs a tool turn with pi's event order", async () => {
		const calls: string[] = [];
		const { loop, events } = fauxLoop(
			[
				fauxAssistantMessage([fauxToolCall('lookup', {}, { id: 'call_1' })], {
					stopReason: 'toolUse',
				}),
				fauxAssistantMessage([fauxText('Done.')]),
			],
			[
				tool('lookup', async (id) => {
					calls.push(id);
					return textResult('found');
				}),
			],
		);

		await loop.prompt('Look it up.');

		expect(calls).toEqual(['call_1']);
		expect(events).toEqual([
			'agent_start',
			'turn_start',
			'message_start:user',
			'message_end:user',
			'message_start:assistant',
			'message_end:assistant',
			'tool_execution_start',
			'tool_execution_end',
			'message_start:toolResult',
			'message_end:toolResult',
			'turn_end',
			'turn_start',
			'message_start:assistant',
			'message_end:assistant',
			'turn_end',
			'agent_end',
		]);
		expect(loop.state.messages.map((message) => message.role)).toEqual([
			'system',
			'user',
			'assistant',
			'toolResult',
			'assistant',
		]);
	});

	it('stops a sequential batch at an abort: the later call never runs', async () => {
		const started = Promise.withResolvers<void>();
		const calls = { first: 0, second: 0 };
		const { loop, recorded } = fauxLoop(
			[
				fauxAssistantMessage(
					[
						fauxToolCall('first', {}, { id: 'call_first' }),
						fauxToolCall('second', {}, { id: 'call_second' }),
					],
					{ stopReason: 'toolUse' },
				),
			],
			[
				tool(
					'first',
					(_id, _args, signal) => {
						calls.first += 1;
						started.resolve();
						return new Promise((_resolve, reject) =>
							signal?.addEventListener('abort', () => reject(signal.reason)),
						);
					},
					{ executionMode: 'sequential' },
				),
				tool('second', async () => {
					calls.second += 1;
					return textResult('second result');
				}),
			],
		);

		const run = loop.prompt('Run both tools.');
		await started.promise;
		loop.abort();
		await run;

		expect(calls).toEqual({ first: 1, second: 0 });
		const batchEnd = recorded.find(
			(event) => event.type === 'turn_end' && event.toolResults.length > 0,
		);
		expect(
			batchEnd?.type === 'turn_end' &&
				batchEnd.toolResults.map((result) => [result.toolCallId, result.isError]),
		).toEqual([['call_first', true]]);
		expect(loop.state.messages.at(-1)).toMatchObject({
			role: 'assistant',
			stopReason: 'aborted',
		});
	});

	it('continues the same run when a message is steered after the model stopped', async () => {
		const { loop, faux, events } = fauxLoop([
			fauxAssistantMessage([fauxText('First answer.')]),
			(context) =>
				fauxAssistantMessage([
					fauxText(`Saw: ${JSON.stringify(context.messages.at(-1)?.content)}`),
				]),
		]);
		loop.subscribe((event) => {
			const isFirstAnswer = event.type === 'turn_end' && faux.state.callCount === 1;
			if (isFirstAnswer)
				loop.steer({
					role: 'user',
					content: [{ type: 'text', text: 'One more thing.' }],
					timestamp: 1,
				});
		});

		await loop.prompt('Hello.');

		expect(events.filter((event) => event === 'agent_start')).toHaveLength(1);
		expect(loop.state.messages.at(-1)).toMatchObject({
			role: 'assistant',
			content: [{ type: 'text', text: expect.stringContaining('One more thing.') }],
		});
	});

	it('ends the run after a batch whose every result asks to terminate', async () => {
		const { loop, faux } = fauxLoop(
			[
				fauxAssistantMessage([fauxToolCall('finish', {}, { id: 'call_1' })], {
					stopReason: 'toolUse',
				}),
				fauxAssistantMessage([fauxText('Must not run.')]),
			],
			[tool('finish', async () => textResult('finished', true))],
		);

		await loop.prompt('Finish.');

		expect(faux.state.callCount).toBe(1);
		expect(loop.state.messages.at(-1)?.role).toBe('toolResult');
	});

	it('keeps a model error as the last assistant and in errorMessage', async () => {
		const { loop } = fauxLoop([
			fauxAssistantMessage([], {
				stopReason: 'error',
				errorMessage: 'upstream server error',
			}),
		]);

		await loop.prompt('Hello.');

		expect(loop.state.messages.at(-1)).toMatchObject({
			role: 'assistant',
			stopReason: 'error',
		});
		expect(loop.state.errorMessage).toContain('upstream server error');
	});
});
