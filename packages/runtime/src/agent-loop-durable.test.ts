import { logMessageStore } from '@tanstack/ai-harness';
import { describe, expect, it } from 'vitest';
import { AgentLoop } from './agent-loop.ts';
import type { AgentMessage, AgentTool, ToolResultMessage } from './llm-types.ts';
import { sqlite } from './node/agent-execution-store.ts';
import type { ConversationStreamStore } from './runtime/conversation-stream-store.ts';
import { createFlueLogStore } from './runtime/harness-log-store.ts';
import { createInstanceHarnessHost } from './runtime/instance-harness-host.ts';
import {
	type FauxContext,
	type FauxModelDefinition,
	type FauxResponseStep,
	fauxAssistantMessage,
	fauxProvider,
	fauxText,
	fauxToolCall,
} from './test-utils/faux.ts';

const identity = { agentName: 'test-agent', instanceId: 'instance-1' };
const path = 'agents/test-agent/instance-1';
const threadId = 'conversation-1';

/** A loop on the instance host of new SQLite stores, on the faux model. */
async function durableLoop(
	responses: FauxResponseStep[],
	options: { tools?: AgentTool[]; models?: FauxModelDefinition[] } = {},
) {
	const adapter = sqlite();
	await adapter.migrate?.();
	const stores = await adapter.connect();
	const streams = stores.conversationStreamStore;
	const host = createInstanceHarnessHost({
		streams,
		submissions: stores.submissionStore,
		path,
		identity,
		ownerId: 'host-a',
	});
	const faux = fauxProvider(options.models ? { models: options.models } : {});
	faux.setResponses(responses);
	const { createAdapter } = faux.provider;
	const model = faux.getModel();
	if (!createAdapter || !model) throw new Error('The faux provider has no adapter or model.');
	const loop = new AgentLoop({
		initialState: {
			systemPrompt: 'Be brief.',
			model,
			tools: options.tools ?? [],
			thinkingLevel: 'off',
		},
		sessionId: threadId,
		createAdapter: async (model) =>
			createAdapter(model, { auth: undefined, promptCacheKey: threadId }),
		durable: { host, threadId, logId: path },
	});
	const messages: AgentMessage[] = [];
	loop.subscribe((event) => {
		if (event.type === 'message_end') messages.push(event.message);
	});
	return { loop, faux, host, streams, messages };
}

/** The transcript of the thread in the instance stream, read by a new reader. */
async function storedTranscript(streams: ConversationStreamStore) {
	const log = createFlueLogStore(streams, {
		producerId: 'reader',
		identityFor: () => identity,
	});
	const messages = await logMessageStore({ store: log, logId: path }).loadThread(threadId);
	return messages.map(({ role, content }) => ({ role, content }));
}

function lookupTool(calls: string[]): AgentTool {
	return {
		name: 'lookup',
		label: 'lookup',
		description: 'Look something up.',
		parameters: { type: 'object', properties: {} },
		execute: async (toolCallId) => {
			calls.push(toolCallId);
			return { content: [{ type: 'text', text: 'found' }], details: {} };
		},
	};
}

/** The text of each user message that a model call got. Images read as `[image]`. */
function userTexts(context: FauxContext) {
	const texts: string[] = [];
	for (const message of context.messages) {
		if (message.role !== 'user') continue;
		const { content } = message;
		texts.push(
			typeof content === 'string'
				? content
				: content.map((block) => (block.type === 'text' ? block.text : '[image]')).join(''),
		);
	}
	return texts;
}

describe('AgentLoop on a durable host', () => {
	it('closes a cut tool call with an error result and calls the model again', async () => {
		const calls: string[] = [];
		const { loop, faux, host, messages } = await durableLoop(
			[
				fauxAssistantMessage([fauxToolCall('lookup', { query: 'x' }, { id: 'call_1' })], {
					stopReason: 'length',
				}),
				fauxAssistantMessage([fauxText('Done.')]),
			],
			{ tools: [lookupTool(calls)] },
		);

		await loop.prompt('Look it up.', undefined, { inputId: 'submission-1' });
		await host.close();

		expect(calls).toEqual([]);
		const results = messages.filter(
			(message): message is ToolResultMessage => message.role === 'toolResult',
		);
		expect(
			results.map(({ toolCallId, isError, content }) => ({ toolCallId, isError, content })),
		).toEqual([
			{
				toolCallId: 'call_1',
				isError: true,
				content: [
					{
						type: 'text',
						text: 'Tool call "lookup" was not executed: the response hit the output token limit, so its arguments may be truncated. Re-issue the tool call with complete arguments.',
					},
				],
			},
		]);
		expect(faux.state.callCount).toBe(2);
		expect(messages.at(-1)).toMatchObject({
			role: 'assistant',
			content: [{ type: 'text', text: 'Done.' }],
		});
	});

	it('gives a model without image input a placeholder for the image of a joined steer', async () => {
		const seen: string[][] = [];
		const steered: AgentMessage = {
			role: 'user',
			content: [
				{ type: 'text', text: 'Look at this.' },
				{ type: 'image', data: 'aGVsbG8=', mimeType: 'image/png' },
			],
			timestamp: 1,
		};
		let loop: AgentLoop | undefined;
		const steerTool: AgentTool = {
			...lookupTool([]),
			execute: async () => {
				loop?.steer(steered, { inputId: 'steer-1' });
				return { content: [{ type: 'text', text: 'found' }], details: {} };
			},
		};
		const durable = await durableLoop(
			[
				fauxAssistantMessage([fauxToolCall('lookup', {}, { id: 'call_1' })], {
					stopReason: 'toolUse',
				}),
				(context) => {
					seen.push(userTexts(context));
					return fauxAssistantMessage([fauxText('I cannot see it.')]);
				},
			],
			{ tools: [steerTool], models: [{ id: 'text-only', input: ['text'] }] },
		);
		loop = durable.loop;

		await loop.prompt('Look it up.', undefined, { inputId: 'submission-1' });
		await durable.host.close();

		expect(durable.faux.state.callCount).toBe(2);
		expect(seen).toEqual([
			['Look it up.', 'Look at this.(image omitted: model does not support images)'],
		]);
		expect(durable.messages.map((message) => message.role)).toEqual([
			'user',
			'assistant',
			'toolResult',
			'user',
			'assistant',
		]);
	});

	it('sends the real user message to the harness transcript', async () => {
		const { loop, host, streams } = await durableLoop([
			fauxAssistantMessage([fauxText('Hi there.')]),
		]);

		await loop.prompt('Hello.', undefined, { inputId: 'submission-1' });
		await host.close();

		expect(await storedTranscript(streams)).toEqual([
			{ role: 'user', content: 'Hello.' },
			{ role: 'assistant', content: 'Hi there.' },
		]);
	});

	it('gives an ephemeral message to the model call and keeps it out of the transcript', async () => {
		const seen: string[][] = [];
		const { loop, host, streams } = await durableLoop([
			(context) => {
				seen.push(userTexts(context));
				return fauxAssistantMessage([fauxText('Noted.')]);
			},
		]);

		await loop.prompt('Hello.', undefined, {
			inputId: 'submission-1',
			ephemeral: [{ role: 'user', content: 'Today is Monday.' }],
		});
		await host.close();

		expect(seen).toEqual([['Hello.', 'Today is Monday.']]);
		expect(await storedTranscript(streams)).toEqual([
			{ role: 'user', content: 'Hello.' },
			{ role: 'assistant', content: 'Noted.' },
		]);
	});
});
