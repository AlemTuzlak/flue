import { logMessageStore } from '@tanstack/ai-harness';
import { describe, expect, it } from 'vitest';
import { createFlueContext } from './client.ts';
import type { HarnessLogRecord } from './conversation-records.ts';
import { buildConversationContext } from './conversation-reducer.ts';
import { ConversationRecordWriter } from './conversation-writer.ts';
import { useModel, usePersistentState, useTool } from './index.ts';
import type { AgentMessage } from './llm-types.ts';
import { sqlite } from './node/agent-execution-store.ts';
import { setProvider } from './providers/registry.ts';
import {
	createAgentSubmissionSessionHandler,
	createDirectAgentSubmissionInput,
	ensureInstanceIdentity,
} from './runtime/agent-submissions.ts';
import { InMemoryAttachmentStore } from './runtime/attachment-store.ts';
import type { ConversationStreamStore } from './runtime/conversation-stream-store.ts';
import { createFlueLogStore } from './runtime/harness-log-store.ts';
import {
	createHostRecordAppend,
	createInstanceHarnessHost,
	projectFlueRecord,
} from './runtime/instance-harness-host.ts';
import { resolveModel } from './runtime/providers.ts';
import { fauxAssistantMessage, fauxProvider, fauxText, fauxToolCall } from './test-utils/faux.ts';

const instanceId = 'instance-1';
const identity = { agentName: 'CountingAgent', instanceId };
const path = 'agents/CountingAgent/instance-1';

function CountingAgent() {
	useModel('faux/model');
	const [, setCount] = usePersistentState('count', 0);
	useTool({
		name: 'count',
		description: 'Count the items.',
		run: () => {
			setCount(3);
			return 'counted';
		},
	});
	return 'Count the items, then answer.';
}

async function openSqliteStores() {
	const adapter = sqlite();
	await adapter.migrate?.();
	const stores = await adapter.connect();
	return { submissions: stores.submissionStore, streams: stores.conversationStreamStore };
}

/**
 * Run one submission of `CountingAgent` on new SQLite stores: a tool call,
 * then an answer. `durable` puts an instance host in the context.
 */
async function runSubmission(durable: boolean) {
	const faux = fauxProvider({ models: [{ id: 'model' }] });
	faux.setResponses([
		fauxAssistantMessage([fauxToolCall('count', {}, { id: 'call_1' })], { stopReason: 'toolUse' }),
		fauxAssistantMessage([fauxText('There are 3 items.')]),
	]);
	setProvider(faux.provider);
	const { streams, submissions } = await openSqliteStores();
	const binding = {
		host: createInstanceHarnessHost({ streams, submissions, path, identity, ownerId: 'host-a' }),
		logId: path,
	};
	const writer = durable
		? await ConversationRecordWriter.overHarness({
				store: streams,
				path,
				identity,
				append: createHostRecordAppend(binding),
			})
		: await ConversationRecordWriter.create({
				store: streams,
				path,
				identity,
				producerId: 'host-a',
			});
	const ctx = createFlueContext({
		id: instanceId,
		agentName: identity.agentName,
		env: {},
		agentConfig: { resolveModel },
		conversationWriter: writer,
		attachmentStore: new InMemoryAttachmentStore(),
		...(durable ? { harnessHost: binding } : {}),
	});
	const { conversationId } = await ensureInstanceIdentity(writer, CountingAgent, undefined);
	const input = await createDirectAgentSubmissionInput({
		agent: identity.agentName,
		id: instanceId,
		message: { kind: 'user', body: 'How many items?' },
	});
	await submissions.admitDirect(input);
	await submissions.markSubmissionCanonicalReady(input.submissionId);
	await submissions.claimSubmission({
		submissionId: input.submissionId,
		attemptId: 'attempt-1',
		ownerId: 'host-a',
		leaseExpiresAt: Date.now() + 30_000,
	});
	await createAgentSubmissionSessionHandler(CountingAgent, input, (session) =>
		session.processSubmissionInput(input, {
			submissionAttempt: { submissionId: input.submissionId, attemptId: 'attempt-1' },
		}),
	)(ctx);
	await binding.host.close();
	return { streams, writer, conversationId, submissionId: input.submissionId };
}

/** The history of the conversation as the reducer rebuilds it: role and text of each message. */
async function storedHistory(writer: ConversationRecordWriter, conversationId: string) {
	const conversation = await writer.getConversation(conversationId);
	if (!conversation) throw new Error('The conversation is missing.');
	return buildConversationContext(conversation).map(summarize);
}

function summarize(message: AgentMessage) {
	if (message.role === 'signal') return { role: message.role, text: message.content };
	if (typeof message.content === 'string') return { role: message.role, text: message.content };
	return {
		role: message.role,
		text: message.content
			.map((block) =>
				block.type === 'text'
					? block.text
					: block.type === 'toolCall'
						? `call ${block.name} ${block.id}`
						: '',
			)
			.join(''),
	};
}

/** The transcript of `threadId` in the instance stream, read by a new reader. */
async function storedTranscript(streams: ConversationStreamStore, threadId: string) {
	const log = createFlueLogStore(streams, { producerId: 'reader', identityFor: () => identity });
	const messages = await logMessageStore({
		store: log,
		logId: path,
		project: { record: projectFlueRecord, version: 'flue-signals-1' },
	}).loadThread(threadId);
	return messages.map(({ role, content, toolCalls, toolCallId }) => ({
		role,
		content,
		...(toolCalls ? { toolCalls: toolCalls.map(({ id, function: call }) => [id, call.name]) } : {}),
		...(toolCallId ? { toolCallId } : {}),
	}));
}

/** The records of each harness batch of the instance stream, in order. */
async function harnessBatches(streams: ConversationStreamStore) {
	const page = await streams.read(path);
	const batches: HarnessLogRecord[][] = [];
	for (const batch of page.batches) {
		for (const record of batch.records) {
			if (record.type === 'harness_log_batch') batches.push(record.records);
		}
	}
	return batches;
}

const expectedHistory = [
	{ role: 'user', text: 'How many items?' },
	{ role: 'assistant', text: 'call count call_1' },
	{ role: 'toolResult', text: '"counted"' },
	{ role: 'assistant', text: 'There are 3 items.' },
];

describe('a Flue session on the instance harness host', () => {
	it('keeps the same Flue history as a session without a host', async () => {
		const today = await runSubmission(false);
		const durable = await runSubmission(true);

		expect(await storedHistory(today.writer, today.conversationId)).toEqual(expectedHistory);
		expect(await storedHistory(durable.writer, durable.conversationId)).toEqual(expectedHistory);
		expect((await durable.writer.loadReducedState()).state.get('count')).toBe(3);
	});

	it('runs the conversation as a harness thread with the same messages', async () => {
		const { streams, conversationId, submissionId } = await runSubmission(true);

		const inputs = (await harnessBatches(streams))
			.flat()
			.filter((record) => record.type === 'harness.input');
		expect(inputs.map((record) => [record.thread, record.inputId])).toEqual([
			[conversationId, submissionId],
		]);

		expect(await storedTranscript(streams, conversationId)).toEqual([
			{ role: 'user', content: 'How many items?' },
			{ role: 'assistant', content: null, toolCalls: [['call_1', 'count']] },
			// TanStack parses a JSON tool result: Flue's `"counted"` is `counted` here.
			{ role: 'tool', content: 'counted', toolCallId: 'call_1' },
			{ role: 'assistant', content: 'There are 3 items.' },
		]);
	});

	it('lands a state write from a tool with the tool outcome, after the tool result and before the answer', async () => {
		const { streams, conversationId } = await runSubmission(true);

		const batches = await harnessBatches(streams);
		const indexOf = (match: (record: HarnessLogRecord) => boolean) =>
			batches.findIndex((records) => records.some(match));
		const writeBatch = indexOf((record) => record.type === 'state_write');
		expect(
			batches[writeBatch]?.map((record) => [record.type, record.thread, record.name ?? null]),
		).toEqual([
			['state_write', conversationId, 'count'],
			['tool_outcome', conversationId, null],
		]);
		const toolResult = indexOf(
			(record) => record.type === 'harness.tool.result' && record.toolCallId === 'call_1',
		);
		const answerStarts = batches
			.map((records, index) => ({ records, index }))
			.filter(({ records }) =>
				records.some((record) => record.type === 'assistant_message_started'),
			)
			.map(({ index }) => index);
		expect(toolResult).toBeGreaterThan(-1);
		expect(writeBatch).toBeGreaterThan(toolResult);
		expect(answerStarts).toHaveLength(2);
		expect(writeBatch).toBeLessThan(answerStarts[1] ?? -1);
	});
});

describe('projectFlueRecord', () => {
	const signal = {
		type: 'signal',
		v: 1,
		thread: 'conv_1',
		messageId: 'entry_signal',
		signalType: 'resources',
		content: 'Tools changed.',
	};

	it('adds a signal record to the model context as a rendered user message', () => {
		expect(
			projectFlueRecord({ messages: [{ role: 'user', content: 'Hi' }], record: signal }),
		).toEqual([
			{ role: 'user', content: 'Hi' },
			{
				role: 'user',
				content: '<signal type="resources">\nTools changed.\n</signal>',
				id: 'entry_signal',
			},
		]);
	});

	it('leaves out the stream recovery pair and the other Flue records', () => {
		const messages = [{ role: 'user' as const, content: 'Hi' }];
		expect(
			projectFlueRecord({ messages, record: { ...signal, signalType: 'stream_interrupted' } }),
		).toBeUndefined();
		expect(
			projectFlueRecord({ messages, record: { ...signal, signalType: 'stream_continued' } }),
		).toBeUndefined();
		expect(
			projectFlueRecord({ messages, record: { type: 'user_message', v: 1, content: [] } }),
		).toBeUndefined();
	});
});

describe('the local runtime', () => {
	function BriefAgent() {
		useModel('faux/model');
		return 'Be brief.';
	}

	it('keeps the conversation on its own durable host across harnesses', async () => {
		const seen: string[][] = [];
		const faux = fauxProvider({ models: [{ id: 'model' }] });
		faux.setResponses([
			fauxAssistantMessage([fauxText('One.')]),
			(context) => {
				seen.push(context.messages.map((message) => summarize(message).text));
				return fauxAssistantMessage([fauxText('Two.')]);
			},
		]);
		setProvider(faux.provider);
		const ctx = createFlueContext({
			id: instanceId,
			agentName: 'BriefAgent',
			env: {},
			agentConfig: { resolveModel },
		});

		const first = await ctx.initializeRootHarness(BriefAgent);
		const firstAnswer = await first.prompt('Hi.');
		const second = await ctx.initializeRootHarness(BriefAgent);
		const secondAnswer = await second.prompt('Again.');

		expect([firstAnswer.text, secondAnswer.text]).toEqual(['One.', 'Two.']);
		expect(seen).toEqual([['Hi.', 'One.', 'Again.']]);
	});
});
