/**
 * Review focus: a database written by today's Flue, with one in-flight
 * submission: the history reads the same, and the submission continues.
 *
 * Each test writes the old database as a fixture: record literals with the
 * shapes that Flue wrote before the harness, as plain batches straight to a
 * SQLite stream store, and the submission row through the submission store.
 * Then the runtime starts on that file.
 */
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, expect, it } from 'vitest';
import { projectAgentConversationSnapshot } from './conversation-public.ts';
import { loadReducedConversationState } from './conversation-reader.ts';
import type { ConversationRecord } from './conversation-records.ts';
import { encodeCanonicalId } from './conversation-records.ts';
import { getActiveConversationPath, toolResultEntryId } from './conversation-reducer.ts';
import { init, useModel, useTool } from './index.ts';
import type { AgentMessage } from './llm-types.ts';
import { sqlite, start } from './node/index.ts';
import { agentStreamPath } from './runtime/stream-offsets.ts';
import {
	type FauxContext,
	fauxAssistantMessage,
	fauxProvider,
	fauxText,
	fauxThinking,
	fauxToolCall,
} from './test-utils/faux.ts';

const AGENT = 'LegacyAgent';
const INSTANCE = 'legacy-1';
const CONVERSATION = 'conv_legacy';
const PATH = agentStreamPath(AGENT, INSTANCE);
const TIME = '2026-01-01T00:00:00.000Z';
/** The model of the faux provider, as Flue records it. */
const MODEL_INFO = { api: 'faux', provider: 'faux', model: 'model' };

let directory: string;
let file: string;

beforeEach(async () => {
	directory = await mkdtemp(join(tmpdir(), 'flue-legacy-stream-'));
	file = join(directory, 'flue.db');
});

afterEach(async () => {
	await rm(directory, { recursive: true, force: true });
});

function LegacyAgent() {
	useModel('faux/model');
	useTool({ name: 'look', description: 'Look around.', run: async () => 'seen it' });
	return 'Reply to the user.';
}

// ─── The fixture: records as Flue wrote them before the harness ─────────────

function envelope(id: string, owner?: { submissionId: string; attemptId: string }) {
	return {
		v: 1 as const,
		id,
		conversationId: CONVERSATION,
		harness: 'default',
		session: 'default',
		timestamp: TIME,
		...owner,
	};
}

function inputEntryId(submissionId: string) {
	return `entry_dispatch_${encodeCanonicalId(submissionId)}`;
}

const created: ConversationRecord = {
	...envelope(`record_conversation_created_${CONVERSATION}`),
	type: 'conversation_created',
	kind: 'root',
	affinityKey: 'aff_legacy',
	createdAt: TIME,
	uid: 'inst_legacy',
};

/**
 * One settled exchange of `sub_old`: `Hello.`, an answer with reasoning,
 * text, and a `look` call, the call's result, and the final answer `Done.`.
 */
function settledExchange(): ConversationRecord[][] {
	const owner = { submissionId: 'sub_old', attemptId: 'attempt_old' };
	const input = inputEntryId('sub_old');
	const outcomeId = 'record_tool_outcome_old_call_look';
	return [
		[
			{
				...envelope('record_dispatch_input_sub_old', owner),
				type: 'user_message',
				messageId: input,
				parentId: null,
				content: [{ type: 'text', text: 'Hello.' }],
			},
		],
		[
			{
				...envelope('record_old_1', owner),
				type: 'assistant_message_started',
				messageId: 'entry_old_answer_1',
				parentId: input,
				modelInfo: MODEL_INFO,
			},
			{
				...envelope('record_old_2', owner),
				type: 'assistant_reasoning_started',
				messageId: 'entry_old_answer_1',
				blockId: 'block_old_1',
				blockIndex: 0,
			},
			{
				...envelope('record_old_3', owner),
				type: 'assistant_reasoning_delta',
				messageId: 'entry_old_answer_1',
				blockId: 'block_old_1',
				sequence: 0,
				delta: 'Think first.',
			},
			{
				...envelope('record_old_4', owner),
				type: 'assistant_reasoning_completed',
				messageId: 'entry_old_answer_1',
				blockId: 'block_old_1',
				deltaCount: 1,
			},
			{
				...envelope('record_old_5', owner),
				type: 'assistant_text_started',
				messageId: 'entry_old_answer_1',
				blockId: 'block_old_2',
				blockIndex: 1,
			},
			{
				...envelope('record_old_6', owner),
				type: 'assistant_text_delta',
				messageId: 'entry_old_answer_1',
				blockId: 'block_old_2',
				sequence: 0,
				delta: 'Let me look.',
			},
			{
				...envelope('record_old_7', owner),
				type: 'assistant_text_completed',
				messageId: 'entry_old_answer_1',
				blockId: 'block_old_2',
				deltaCount: 1,
			},
			{
				...envelope('record_old_8', owner),
				type: 'assistant_tool_call',
				messageId: 'entry_old_answer_1',
				blockId: 'block_old_3',
				blockIndex: 2,
				toolCallId: 'call_look',
				name: 'look',
				arguments: {},
			},
			{
				...envelope('record_old_9', owner),
				type: 'assistant_message_completed',
				messageId: 'entry_old_answer_1',
				stopReason: 'toolUse',
				usage: zeroUsage(),
			},
		],
		[
			{
				...envelope(outcomeId, owner),
				type: 'tool_outcome',
				assistantMessageId: 'entry_old_answer_1',
				toolCallId: 'call_look',
				toolName: 'look',
				isError: false,
				content: [{ type: 'text', text: '"seen it"' }],
				output: 'seen it',
			},
			{
				...envelope('record_old_10', owner),
				type: 'tool_results_committed',
				assistantMessageId: 'entry_old_answer_1',
				parentId: 'entry_old_answer_1',
				outcomeIds: [outcomeId],
			},
		],
		[
			{
				...envelope('record_old_11', owner),
				type: 'assistant_message_started',
				messageId: 'entry_old_answer_2',
				parentId: toolResultEntryId('entry_old_answer_1', 'call_look'),
				modelInfo: MODEL_INFO,
			},
			{
				...envelope('record_old_12', owner),
				type: 'assistant_text_started',
				messageId: 'entry_old_answer_2',
				blockId: 'block_old_4',
				blockIndex: 0,
			},
			{
				...envelope('record_old_13', owner),
				type: 'assistant_text_delta',
				messageId: 'entry_old_answer_2',
				blockId: 'block_old_4',
				sequence: 0,
				delta: 'Done.',
			},
			{
				...envelope('record_old_14', owner),
				type: 'assistant_text_completed',
				messageId: 'entry_old_answer_2',
				blockId: 'block_old_4',
				deltaCount: 1,
			},
			{
				...envelope('record_old_15', owner),
				type: 'assistant_message_completed',
				messageId: 'entry_old_answer_2',
				stopReason: 'stop',
				usage: zeroUsage(),
			},
		],
	];
}

function zeroUsage() {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

/**
 * Write the old database: the conversation and the settled exchange of
 * `sub_old`, then, with `inFlight`, the running submission `sub_live` (its
 * row claimed by a host that is gone) and the batches it wrote. Each
 * submission's row goes through the submission store, as Flue's ledger
 * moved it.
 */
async function writeLegacyDatabase(inFlight?: {
	body: string;
	batches(owner: { submissionId: string; attemptId: string }): ConversationRecord[][];
}) {
	const database = sqlite(file);
	await database.migrate?.();
	const stores = await database.connect();
	const streams = stores.conversationStreamStore;
	const submissions = stores.submissionStore;
	await streams.createStream(PATH, { agentName: AGENT, instanceId: INSTANCE });
	const claim = await streams.acquireProducer(PATH, 'old-flue');
	let sequence = claim.nextProducerSequence;
	const append = async (
		records: ConversationRecord[],
		submission?: { submissionId: string; attemptId: string },
	) => {
		await streams.append({
			path: PATH,
			producerId: claim.producerId,
			producerEpoch: claim.producerEpoch,
			incarnation: claim.incarnation,
			producerSequence: sequence,
			...(submission ? { submission } : {}),
			records,
		});
		sequence += 1;
	};
	const run = async (
		owner: { submissionId: string; attemptId: string },
		body: string,
		leaseExpiresAt: number,
	) => {
		await submissions.admitDispatch({
			submissionId: owner.submissionId,
			agent: AGENT,
			id: INSTANCE,
			message: { kind: 'user', body },
			acceptedAt: TIME,
		});
		await submissions.markSubmissionCanonicalReady(owner.submissionId);
		await submissions.claimSubmission({ ...owner, ownerId: 'gone-host', leaseExpiresAt });
		await submissions.markSubmissionInputApplied(owner);
	};

	await append([created]);
	const old = { submissionId: 'sub_old', attemptId: 'attempt_old' };
	await run(old, 'Hello.', Date.now() + 60_000);
	for (const batch of settledExchange()) await append(batch, old);
	const settlement: ConversationRecord = {
		...envelope('record_submission_settled_sub_old', old),
		type: 'submission_settled',
		submissionId: old.submissionId,
		outcome: 'completed',
	};
	await submissions.reserveSubmissionSettlement(old, {
		recordId: settlement.id,
		record: settlement,
	});
	await append([settlement], old);
	await submissions.finalizeSubmissionSettlement(old, settlement.id);

	if (inFlight) {
		const live = { submissionId: 'sub_live', attemptId: 'attempt_gone' };
		await run(live, inFlight.body, Date.now() - 1_000);
		for (const batch of inFlight.batches(live)) await append(batch, live);
	}
	await database.close?.();
}

/** The user message of `sub_live`. */
function liveInput(owner: { submissionId: string; attemptId: string }, body: string) {
	return {
		...envelope('record_dispatch_input_sub_live', owner),
		type: 'user_message' as const,
		messageId: inputEntryId('sub_live'),
		parentId: 'entry_old_answer_2',
		content: [{ type: 'text' as const, text: body }],
	};
}

// ─── Reading the database ───────────────────────────────────────────────────

async function readDatabase<T>(read: (stores: Awaited<ReturnType<typeof connect>>) => Promise<T>) {
	const database = sqlite(file);
	try {
		return await read(await connect(database));
	} finally {
		await database.close?.();
	}
}

async function connect(database: ReturnType<typeof sqlite>) {
	return database.connect();
}

/** The public history of the conversation: its projected messages. */
function readHistory() {
	return readDatabase(async (stores) => {
		const state = await loadReducedConversationState({
			store: stores.conversationStreamStore,
			path: PATH,
		});
		return projectAgentConversationSnapshot(state)?.messages ?? [];
	});
}

/** The active path of the conversation: role, and the text or signal type of each message. */
function readPath() {
	return readDatabase(async (stores) => {
		const state = await loadReducedConversationState({
			store: stores.conversationStreamStore,
			path: PATH,
		});
		const conversation = state.conversations.get(CONVERSATION);
		if (!conversation) return [];
		return getActiveConversationPath(conversation).flatMap((entry) =>
			entry.type === 'message' ? [summary(entry.message)] : [],
		);
	});
}

/** The seed records in the stream. */
function readSeedRecords() {
	return readDatabase(async (stores) => {
		const page = await stores.conversationStreamStore.read(PATH);
		return page.batches.flatMap((batch) =>
			batch.records.flatMap((record) =>
				record.type === 'harness_log_batch'
					? record.records.filter((inner) => inner.type === 'flue.seed')
					: [],
			),
		);
	});
}

function summary(message: AgentMessage) {
	switch (message.role) {
		case 'signal':
			return `signal:${message.type}`;
		case 'assistant':
			return `assistant(${message.stopReason}):${message.content
				.map((block) => (block.type === 'text' ? block.text : block.type))
				.join('|')}`;
		case 'toolResult':
			return `toolResult:${message.toolCallId}`;
		case 'user':
			return `user:${
				typeof message.content === 'string'
					? message.content
					: message.content.map((block) => (block.type === 'text' ? block.text : block.type))
			}`;
		case 'system':
			return 'system';
	}
}

/** The messages of a model request, without their timestamps. */
function requestMessages(context: FauxContext) {
	return context.messages.map(({ timestamp: _timestamp, ...message }) => message);
}

/** Start the runtime on the database file. */
function startRuntime(faux: ReturnType<typeof fauxProvider>) {
	return start({ agents: [LegacyAgent], db: sqlite(file), providers: [faux.provider], env: {} });
}

/** Answers that record the request of each model call. */
function recordingProvider(answers: string[]) {
	const faux = fauxProvider({ models: [{ id: 'model', reasoning: true }] });
	const requests: FauxContext[] = [];
	faux.setResponses(
		answers.map((answer) => (context: FauxContext) => {
			requests.push(context);
			return fauxAssistantMessage([fauxText(answer)]);
		}),
	);
	return { faux, requests };
}

// ─── Tests ──────────────────────────────────────────────────────────────────

it('reads an old history the same, and a new prompt sees it as the durable path stores it', async () => {
	// The durable path from the start: the same exchange, then the new prompt.
	const durableFile = join(directory, 'durable.db');
	const durable = fauxProvider({ models: [{ id: 'model', reasoning: true }] });
	const durableRequests: FauxContext[] = [];
	durable.setResponses([
		fauxAssistantMessage(
			[
				fauxThinking('Think first.'),
				fauxText('Let me look.'),
				fauxToolCall('look', {}, { id: 'call_look' }),
			],
			{ stopReason: 'toolUse' },
		),
		fauxAssistantMessage([fauxText('Done.')]),
		(context) => {
			durableRequests.push(context);
			return fauxAssistantMessage([fauxText('Again done.')]);
		},
	]);
	const durableRuntime = await start({
		agents: [LegacyAgent],
		db: sqlite(durableFile),
		providers: [durable.provider],
		env: {},
	});
	try {
		const agent = init(LegacyAgent, { id: INSTANCE });
		await agent.read(await agent.dispatch('Hello.'));
		await agent.read(await agent.dispatch('Again.'));
	} finally {
		await durableRuntime.stop();
	}

	await writeLegacyDatabase();
	const before = await readHistory();
	const { faux, requests } = recordingProvider(['Again done.']);
	const runtime = await startRuntime(faux);
	try {
		const agent = init(LegacyAgent, { id: INSTANCE });
		await expect(agent.read(await agent.dispatch('Again.'))).resolves.toMatchObject({
			text: 'Again done.',
		});
	} finally {
		await runtime.stop();
	}

	const after = await readHistory();
	expect(after.slice(0, before.length)).toEqual(before);
	expect(after.map((message) => message.role)).toEqual(['user', 'assistant', 'user', 'assistant']);
	// The request holds the old messages as a durable-path transcript does:
	// the model's answers name the TanStack adapter as their source.
	const answerOf = (content: unknown[], stopReason: string) => ({
		role: 'assistant',
		content,
		api: 'text',
		provider: 'fake',
		model: 'model',
		usage: zeroUsage(),
		stopReason,
	});
	expect(requests).toHaveLength(1);
	const request = requestMessages(requests[0] ?? { messages: [] });
	expect(request).toEqual([
		{ role: 'user', content: 'Hello.' },
		answerOf(
			[
				{ type: 'thinking', thinking: 'Think first.' },
				{ type: 'text', text: 'Let me look.' },
				{ type: 'toolCall', id: 'call_look', name: 'look', arguments: {} },
			],
			'toolUse',
		),
		{
			role: 'toolResult',
			toolCallId: 'call_look',
			toolName: '',
			content: [{ type: 'text', text: '"seen it"' }],
			isError: false,
		},
		answerOf([{ type: 'text', text: 'Done.' }], 'stop'),
		{ role: 'user', content: 'Again.' },
	]);
	// Each message that the durable path sends for the same exchange goes out the same.
	expect(durableRequests).toHaveLength(1);
	for (const message of requestMessages(durableRequests[0] ?? { messages: [] })) {
		expect(request).toContainEqual(message);
	}
});

it('continues an in-flight submission whose input applied, with no new user message', async () => {
	await writeLegacyDatabase({
		body: 'What now?',
		batches: (owner) => [[liveInput(owner, 'What now?')]],
	});
	const { faux, requests } = recordingProvider(['This now.']);
	const runtime = await startRuntime(faux);
	try {
		await expect(init(LegacyAgent, { id: INSTANCE }).read('sub_live')).resolves.toMatchObject({
			text: 'This now.',
		});
	} finally {
		await runtime.stop();
	}

	expect(requests).toHaveLength(1);
	expect(requests[0]?.messages.map(summary)).toEqual([
		'user:Hello.',
		'assistant(toolUse):thinking|Let me look.|toolCall',
		'toolResult:call_look',
		'assistant(stop):Done.',
		'user:What now?',
	]);
	expect(await readPath()).toEqual([
		'user:Hello.',
		'assistant(toolUse):thinking|Let me look.|toolCall',
		'toolResult:call_look',
		'assistant(stop):Done.',
		'user:What now?',
		'assistant(stop):This now.',
	]);
	await readDatabase(async (stores) => {
		await expect(stores.submissionStore.getSubmission('sub_live')).resolves.toMatchObject({
			status: 'settled',
			attemptCount: 2,
		});
	});
});

it('keeps a partial streamed answer and continues it after the two recovery signals', async () => {
	await writeLegacyDatabase({
		body: 'Write a poem.',
		batches: (owner) => [
			[liveInput(owner, 'Write a poem.')],
			[
				{
					...envelope('record_live_1', owner),
					type: 'assistant_message_started',
					messageId: 'entry_live_answer',
					parentId: inputEntryId('sub_live'),
					modelInfo: MODEL_INFO,
				},
				{
					...envelope('record_live_2', owner),
					type: 'assistant_text_started',
					messageId: 'entry_live_answer',
					blockId: 'block_live_1',
					blockIndex: 0,
				},
				{
					...envelope('record_live_3', owner),
					type: 'assistant_text_delta',
					messageId: 'entry_live_answer',
					blockId: 'block_live_1',
					sequence: 0,
					delta: 'Roses are',
				},
			],
		],
	});
	const { faux, requests } = recordingProvider([' red.']);
	const runtime = await startRuntime(faux);
	try {
		await init(LegacyAgent, { id: INSTANCE }).read('sub_live');
	} finally {
		await runtime.stop();
	}

	expect(requests).toHaveLength(1);
	expect(requests[0]?.messages.slice(-4)).toMatchObject([
		{ role: 'user', content: 'Write a poem.' },
		{ role: 'assistant', content: [{ type: 'text', text: 'Roses are' }] },
		{
			role: 'user',
			content:
				'<signal type="stream_interrupted">\nThe previous assistant stream was interrupted.\n</signal>',
		},
		{
			role: 'user',
			content:
				'<signal type="stream_continued">\nContinue from the durable partial assistant response.\n</signal>',
		},
	]);
	expect((await readPath()).slice(4)).toEqual([
		'user:Write a poem.',
		'assistant(aborted):Roses are',
		'signal:stream_interrupted',
		'signal:stream_continued',
		'assistant(stop): red.',
	]);
	await readDatabase(async (stores) => {
		await expect(stores.submissionStore.getSubmission('sub_live')).resolves.toMatchObject({
			status: 'settled',
			attemptCount: 2,
		});
	});
});

it('writes the seed once', async () => {
	await writeLegacyDatabase();
	const first = recordingProvider(['First.']);
	const runtimeA = await startRuntime(first.faux);
	try {
		const agent = init(LegacyAgent, { id: INSTANCE });
		await agent.read(await agent.dispatch('One.'));
	} finally {
		await runtimeA.stop();
	}
	expect(await readSeedRecords()).toHaveLength(1);

	const second = recordingProvider(['Second.']);
	const runtimeB = await startRuntime(second.faux);
	try {
		const agent = init(LegacyAgent, { id: INSTANCE });
		await agent.read(await agent.dispatch('Two.'));
	} finally {
		await runtimeB.stop();
	}

	expect(await readSeedRecords()).toHaveLength(1);
	expect(second.requests[0]?.messages.map(summary)).toEqual([
		'user:Hello.',
		'assistant(toolUse):thinking|Let me look.|toolCall',
		'toolResult:call_look',
		'assistant(stop):Done.',
		'user:One.',
		'assistant(stop):First.',
		'user:Two.',
	]);
});
