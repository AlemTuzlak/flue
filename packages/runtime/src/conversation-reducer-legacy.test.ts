import { describe, expect, it } from 'vitest';
import {
	type ConversationRecord,
	flueRecordsOf,
	type HarnessLogBatchRecord,
	type HarnessLogRecord,
} from './conversation-records.ts';
import { createReducedInstanceState, reduceConversationRecords } from './conversation-reducer.ts';

const timestamp = '2026-10-06T00:00:00.000Z';
const logId = 'agents/support/instance-1';
const envelope = {
	v: 1 as const,
	conversationId: 'conv_root',
	harness: 'default',
	session: 'default',
	timestamp,
};
const usage = {
	input: 10,
	output: 5,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 15,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

function userMessage(
	messageId: string,
	parentId: string | null,
	submissionId: string,
	text: string,
) {
	return {
		...envelope,
		submissionId,
		id: `record_${messageId}`,
		type: 'user_message',
		messageId,
		parentId,
		content: [{ type: 'text', text }],
	} satisfies ConversationRecord;
}

/** An assistant message that streams one text block. `completed: false` stops after the first delta. */
function assistantRecords(options: {
	messageId: string;
	parentId: string;
	submissionId: string;
	text: string;
	completed: boolean;
}): ConversationRecord[] {
	const shared = {
		...envelope,
		submissionId: options.submissionId,
		messageId: options.messageId,
	};
	const blockId = `block_${options.messageId}`;
	const streamed: ConversationRecord[] = [
		{
			...shared,
			id: `record_${options.messageId}_started`,
			type: 'assistant_message_started',
			parentId: options.parentId,
			modelInfo: {
				api: 'anthropic-messages',
				provider: 'anthropic',
				model: 'claude-sonnet-4-5',
			},
		},
		{
			...shared,
			id: `record_${options.messageId}_text`,
			type: 'assistant_text_started',
			blockId,
			blockIndex: 0,
		},
		{
			...shared,
			id: `record_${options.messageId}_delta`,
			type: 'assistant_text_delta',
			blockId,
			sequence: 0,
			delta: options.text,
		},
	];
	if (!options.completed) return streamed;
	return [
		...streamed,
		{
			...shared,
			id: `record_${options.messageId}_text_done`,
			type: 'assistant_text_completed',
			blockId,
			deltaCount: 1,
		},
		{
			...shared,
			id: `record_${options.messageId}_done`,
			type: 'assistant_message_completed',
			stopReason: 'stop',
			usage,
		},
	];
}

/** Today's stream: a settled submission, a state write, and a submission still in flight. */
const legacyBatches: ConversationRecord[][] = [
	[
		{
			...envelope,
			id: 'record_created',
			type: 'conversation_created',
			kind: 'root',
			affinityKey: 'affinity_root',
			createdAt: timestamp,
			uid: 'uid_1',
		},
		userMessage('entry_user_a', null, 'sub_a', 'First question.'),
	],
	assistantRecords({
		messageId: 'entry_assistant_a',
		parentId: 'entry_user_a',
		submissionId: 'sub_a',
		text: 'First answer.',
		completed: true,
	}),
	[
		{
			...envelope,
			id: 'record_settled_a',
			type: 'submission_settled',
			submissionId: 'sub_a',
			outcome: 'completed',
		},
		{
			...envelope,
			id: 'record_state',
			type: 'state_write',
			name: 'count',
			value: 1,
		},
	],
	[
		userMessage('entry_user_b', 'entry_assistant_a', 'sub_b', 'Second question.'),
		...assistantRecords({
			messageId: 'entry_assistant_b',
			parentId: 'entry_user_b',
			submissionId: 'sub_b',
			text: 'Partial',
			completed: false,
		}),
	],
];

/** Records the harness and TanStack write next to Flue's records. Flue's fold skips them. */
function harnessNoise(seq: number): HarnessLogRecord[] {
	return [
		{
			type: 'harness.input',
			inputId: `input_${seq}`,
			input: { op: 'prompt', message: 'Hi' },
			at: 1,
		},
		{
			type: 'harness.event',
			operationId: 'op_1',
			event: { type: 'TEXT_MESSAGE_CONTENT', delta: 'x' },
		},
		{
			type: 'harness.transcript',
			keep: 0,
			add: [{ role: 'user', content: 'Hi' }],
		},
		{ type: 'tanstack.compaction', reason: 'threshold', head: [], from: 0 },
	];
}

/** The same Flue records, each batch written as one harness append with the harness's `thread` field. */
function asHarnessBatches(batches: ConversationRecord[][]) {
	let seq = 1;
	return batches.map((records, index) => {
		const inner: HarnessLogRecord[] = [
			...harnessNoise(index),
			...records.map((record) => ({ ...record, thread: logId })),
		];
		const batch: HarnessLogBatchRecord = {
			v: 1,
			id: `record_harness_${index}`,
			type: 'harness_log_batch',
			conversationId: '',
			harness: '',
			session: '',
			timestamp,
			seq,
			records: inner,
		};
		seq += inner.length;
		return [batch];
	});
}

function fold(batches: ConversationRecord[][]) {
	return batches.reduce(
		(state, records, index) => reduceConversationRecords(state, records, String(index)),
		createReducedInstanceState(),
	);
}

describe('a stream that moves to harness batches', () => {
	it('folds to the same state as the same records written directly', () => {
		const legacy = fold(legacyBatches);
		const harness = fold(asHarnessBatches(legacyBatches));

		const root = legacy.conversations.get('conv_root');
		expect([...(root?.entries.keys() ?? [])]).toEqual([
			'entry_user_a',
			'entry_assistant_a',
			'entry_user_b',
		]);
		expect([...(root?.inProgressMessages.keys() ?? [])]).toEqual(['entry_assistant_b']);
		expect(harness).toEqual(legacy);
	});

	it('reads the Flue records of a mixed stream, old batches first', () => {
		const [oldBatch, ...rest] = legacyBatches;
		const harness = fold([oldBatch ?? [], ...asHarnessBatches(rest)]);

		expect(harness).toEqual(fold(legacyBatches));
	});
});

describe('flueRecordsOf', () => {
	it('gives the Flue records of a harness batch in order, without the harness fields and records', () => {
		const [[batch] = []] = asHarnessBatches([
			[userMessage('entry_user_a', null, 'sub_a', 'Hello.')],
		]);

		expect(flueRecordsOf(batch ? [batch] : [])).toEqual([
			userMessage('entry_user_a', null, 'sub_a', 'Hello.'),
		]);
	});

	it('gives the records of an old batch as they are', () => {
		const records = legacyBatches[0] ?? [];

		expect(flueRecordsOf(records)).toEqual(records);
	});
});
