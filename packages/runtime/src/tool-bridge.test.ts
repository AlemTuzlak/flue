import { chat, type ModelMessage } from '@tanstack/ai';
import { fakeText } from '@tanstack/ai/testing';
import { createHarnessHost, defineHarness } from '@tanstack/ai-harness';
import { defineAIPersistence } from '@tanstack/ai-persistence';
import { describe, expect, it } from 'vitest';
import type { AgentTool, AgentToolResult } from './llm-types.ts';
import { sqlite } from './node/agent-execution-store.ts';
import { InMemoryConversationStreamStore } from './runtime/conversation-stream-store.ts';
import { createFlueLeaseStore } from './runtime/harness-lease-store.ts';
import { createFlueLogStore } from './runtime/harness-log-store.ts';
import { type HarnessToolCall, toHarnessTool } from './tool-bridge.ts';
import type { ToolStep } from './tool-types.ts';

function agentTool(
	execute: (toolCallId: string, signal?: AbortSignal) => Promise<AgentToolResult>,
	overrides: Partial<AgentTool> = {},
): AgentTool {
	return {
		name: 'work',
		label: 'Work',
		description: 'Do the work.',
		parameters: { type: 'object', properties: {} },
		execute: (toolCallId, _params, signal) => execute(toolCallId, signal),
		...overrides,
	};
}

/** Runs one tool call through `chat()` and returns the tool message of the next model request. */
async function toolMessageAfter(
	tool: AgentTool,
	options: Parameters<typeof toHarnessTool>[1] = {},
) {
	const requests: ModelMessage[][] = [];
	const fake = fakeText();
	fake.setResponses([
		{ toolCalls: [{ id: 'call-1', name: 'work', input: {} }] },
		({ request }) => {
			requests.push(request.messages);
			return { text: 'Done.' };
		},
	]);
	const stream = chat({
		adapter: fake,
		messages: [{ role: 'user', content: 'Go.' }],
		tools: [toHarnessTool(tool, options)],
	});
	for await (const _chunk of stream) {
		// Drain the run.
	}
	return requests[0]?.find((message) => message.role === 'tool');
}

describe('toHarnessTool', () => {
	it('sends one text block as a string', async () => {
		const tool = agentTool(async () => ({
			content: [{ type: 'text', text: 'hello' }],
			details: undefined,
		}));

		expect((await toolMessageAfter(tool))?.content).toBe('hello');
	});

	it('sends text and images as content parts', async () => {
		const tool = agentTool(async () => ({
			content: [
				{ type: 'text', text: 'Here:' },
				{ type: 'image', data: 'aW1n', mimeType: 'image/png' },
			],
			details: undefined,
		}));

		expect((await toolMessageAfter(tool))?.content).toEqual([
			{ type: 'text', content: 'Here:' },
			{
				type: 'image',
				source: { type: 'data', value: 'aW1n', mimeType: 'image/png' },
			},
		]);
	});

	it('gives the session the call before it runs, and the whole result after', async () => {
		const calls: string[] = [];
		const results: [string, AgentToolResult][] = [];
		const tool = agentTool(async () => ({
			content: [{ type: 'text', text: 'done' }],
			details: { customTool: 'work', output: 1 },
			terminate: true,
		}));

		await toolMessageAfter(tool, {
			onCall: (call) => calls.push(call.toolCallId),
			onResult: (toolCallId, result) => results.push([toolCallId, result]),
		});

		expect(calls).toEqual(['call-1']);
		expect(results).toEqual([
			[
				'call-1',
				{
					content: [{ type: 'text', text: 'done' }],
					details: { customTool: 'work', output: 1 },
					terminate: true,
				},
			],
		]);
	});

	it('gives the model a tool error when the tool throws', async () => {
		const tool = agentTool(async () => {
			throw new Error('Disk is full.');
		});

		const message = await toolMessageAfter(tool);

		expect(message).toMatchObject({
			content: '{"error":"Disk is full."}',
			error: 'Disk is full.',
		});
	});

	it("keeps Flue's step rule: one step name once per call", async () => {
		let step: ToolStep | undefined;
		const tool = agentTool(async () => {
			await step?.do('charge', () => 1);
			await step?.do('charge', () => 2);
			return {
				content: [{ type: 'text', text: 'unreachable' }],
				details: undefined,
			};
		});

		const message = await toolMessageAfter(tool, {
			onCall: (call) => (step = call.step),
		});

		expect(message?.error).toContain('ran step.do("charge") twice in one call');
	});
});

describe('a durable bridged tool, cut by a crash', () => {
	it('runs again, and its finished step returns the stored value', async () => {
		const persistenceStores = sqlite();
		await persistenceStores.migrate?.();
		const { submissionStore } = await persistenceStores.connect();
		const streams = new InMemoryConversationStreamStore();
		let started = () => {};
		const startedOnce = new Promise<void>((resolve) => {
			started = resolve;
		});
		const steps = new Map<string, ToolStep>();
		let executions = 0;
		let charges = 0;
		const tool = agentTool(
			async (toolCallId, signal) => {
				executions += 1;
				const chargeId = await steps.get(toolCallId)?.do('charge', () => {
					charges += 1;
					return 'charge-1';
				});
				if (executions === 1) {
					started();
					await new Promise((_, reject) =>
						signal?.addEventListener('abort', () => reject(signal.reason)),
					);
				}
				return {
					content: [{ type: 'text', text: String(chargeId) }],
					details: undefined,
				};
			},
			{ replay: 'safe' },
		);
		const bridged = toHarnessTool(tool, {
			onCall: (call: HarnessToolCall) => steps.set(call.toolCallId, call.step),
		});
		const fake = fakeText();
		fake.setResponses([
			{ toolCalls: [{ id: 'call-1', name: 'work', input: {} }] },
			{ text: 'Done.' },
		]);
		const harness = defineHarness({
			name: 'flue/tool-bridge-test',
			adapter: fake,
			tools: [bridged],
		});
		const hostOn = (ownerId: string) =>
			createHarnessHost({
				persistence: defineAIPersistence({
					stores: {
						log: createFlueLogStore(streams, {
							producerId: ownerId,
							identityFor: () => ({
								agentName: 'test-agent',
								instanceId: 'instance-1',
							}),
						}),
						leases: createFlueLeaseStore(submissionStore, ownerId),
					},
				}),
			});

		const hostA = hostOn('host-a');
		const sessionA = await hostA.open(harness, {
			threadId: 'agents/test-agent',
		});
		Promise.resolve(sessionA.prompt('Charge it.', { inputId: 'input-1' })).catch(() => undefined);
		await startedOnce;
		const hostB = hostOn('host-b');
		const sessionB = await hostB.open(harness, {
			threadId: 'agents/test-agent',
		});
		const settlement = await sessionB.settled('input-1');
		await hostB.close();
		await hostA.close().catch(() => undefined);

		expect(settlement.outcome).toBe('completed');
		expect({ executions, charges }).toEqual({ executions: 2, charges: 1 });
	});
});
