import { expect, it } from 'vitest';
import { init, useModel, useSandbox } from './index.ts';
import type { AgentMessage } from './llm-types.ts';
import { local, sqlite, start } from './node/index.ts';
import {
	type FauxResponseStep,
	fauxAssistantMessage,
	fauxProvider,
	fauxText,
	fauxToolCall,
} from './test-utils/faux.ts';

/** The text of each user message the model saw, in order. */
function userTexts(messages: readonly AgentMessage[]) {
	return messages.flatMap((message) => {
		if (message.role !== 'user') return [];
		if (typeof message.content === 'string') return [message.content];
		return [message.content.map((block) => (block.type === 'text' ? block.text : '')).join('')];
	});
}

/**
 * An agent whose `wait` tool runs until the test releases it or the run is
 * aborted, on the faux model with `responses`.
 */
async function startWaitingAgent(id: string, responses: FauxResponseStep[]) {
	const toolStarted = Promise.withResolvers<void>();
	const release = Promise.withResolvers<void>();
	function JoinAgent() {
		useModel('faux/model');
		useSandbox({
			...local(),
			tools: () => [
				{
					name: 'wait',
					label: 'Wait',
					description: 'Wait until the test releases the tool.',
					parameters: { type: 'object', properties: {} },
					async execute(_id, _args, signal) {
						toolStarted.resolve();
						await new Promise<void>((resolve, reject) => {
							release.promise.then(resolve);
							signal?.addEventListener('abort', () => reject(signal.reason), {
								once: true,
							});
						});
						return {
							details: {},
							content: [{ type: 'text' as const, text: 'waited' }],
						};
					},
				},
			],
		});
		return 'Answer every message.';
	}
	const faux = fauxProvider({ models: [{ id: 'model' }] });
	faux.setResponses(responses);
	const runtime = await start({
		agents: [JoinAgent],
		db: sqlite(),
		providers: [faux.provider],
		env: {},
	});
	return {
		agent: init(JoinAgent, { id }),
		runtime,
		toolStarted: toolStarted.promise,
		release: () => release.resolve(),
	};
}

it('joins deliveries that arrive during a tool call into the running turn, in admission order', async () => {
	const seen: AgentMessage[][] = [];
	const { agent, runtime, toolStarted, release } = await startWaitingAgent('joins-in-order', [
		fauxAssistantMessage([fauxToolCall('wait', {}, { id: 'call_wait' })], {
			stopReason: 'toolUse',
		}),
		(context) => {
			seen.push(context.messages);
			return fauxAssistantMessage([fauxText('Answered all three.')]);
		},
	]);
	try {
		const first = await agent.dispatch('First.');
		await toolStarted;
		const second = await agent.dispatch('Second.');
		const third = await agent.dispatch('Third.');
		release();

		await expect(agent.read(first)).resolves.toMatchObject({
			text: 'Answered all three.',
		});
		await expect(agent.read(second)).resolves.toMatchObject({
			text: 'Answered all three.',
		});
		await expect(agent.read(third)).resolves.toMatchObject({
			text: 'Answered all three.',
		});
		const texts = userTexts(seen[0] ?? []);
		const at = (text: string) => texts.findIndex((entry) => entry.includes(text));
		expect(at('First.')).toBeGreaterThanOrEqual(0);
		expect(at('Second.')).toBeGreaterThan(at('First.'));
		expect(at('Third.')).toBeGreaterThan(at('Second.'));
	} finally {
		await agent.abort();
		await runtime.stop();
	}
});

it('runs a delivery admitted after an abort request later, on its own', async () => {
	const seen: AgentMessage[][] = [];
	const { agent, runtime, toolStarted } = await startWaitingAgent('abort-then-later', [
		fauxAssistantMessage([fauxToolCall('wait', {}, { id: 'call_wait' })], {
			stopReason: 'toolUse',
		}),
		(context) => {
			seen.push(context.messages);
			return fauxAssistantMessage([fauxText('Answered the later message.')]);
		},
	]);
	try {
		const head = await agent.dispatch('Head.');
		await toolStarted;
		const queued = await agent.dispatch('Queued before the abort.');
		await agent.abort();
		const later = await agent.dispatch('Sent after the abort.');

		await expect(agent.read(head)).rejects.toMatchObject({ outcome: 'aborted' });
		await expect(agent.read(queued)).rejects.toMatchObject({
			outcome: 'aborted',
		});
		await expect(agent.read(later)).resolves.toMatchObject({
			text: 'Answered the later message.',
		});
		// The later delivery ran as its own turn: the aborted delivery never reached the model.
		const texts = userTexts(seen[0] ?? []);
		expect(texts.some((text) => text.includes('Sent after the abort.'))).toBe(true);
		expect(texts.some((text) => text.includes('Queued before the abort.'))).toBe(false);
	} finally {
		await agent.abort();
		await runtime.stop();
	}
});
