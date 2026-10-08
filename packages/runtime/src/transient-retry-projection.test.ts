import { describe, expect, it } from 'vitest';
import {
	projectAgentConversationBatch,
	projectAgentConversationSnapshot,
} from './conversation-public.ts';
import type { ConversationRecord } from './conversation-records.ts';
import { createReducedInstanceState, reduceConversationRecords } from './conversation-reducer.ts';
import { replyFromSnapshot } from './runtime/conversation-observer.ts';

const timestamp = '2026-01-01T00:00:00.000Z';
const usage = {
	input: 10,
	output: 10,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 20,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};
const envelope = {
	v: 1 as const,
	conversationId: 'conv_retry',
	harness: 'default',
	session: 'default',
	timestamp,
	submissionId: 'sub_retry',
};

function assistantRecords(options: {
	messageId: string;
	parentId: string;
	text: string;
	stopReason: 'stop' | 'error';
}): ConversationRecord[] {
	const blockId = `block_${options.messageId}`;
	return [
		{
			...envelope,
			id: `record_${options.messageId}_started`,
			type: 'assistant_message_started',
			messageId: options.messageId,
			parentId: options.parentId,
			modelInfo: { api: 'openai-responses', provider: 'openai', model: 'test-model' },
		},
		{
			...envelope,
			id: `record_${options.messageId}_text_started`,
			type: 'assistant_text_started',
			messageId: options.messageId,
			blockId,
			blockIndex: 0,
		},
		{
			...envelope,
			id: `record_${options.messageId}_delta`,
			type: 'assistant_text_delta',
			messageId: options.messageId,
			blockId,
			sequence: 0,
			delta: options.text,
		},
		{
			...envelope,
			id: `record_${options.messageId}_text_completed`,
			type: 'assistant_text_completed',
			messageId: options.messageId,
			blockId,
			deltaCount: 1,
		},
		{
			...envelope,
			id: `record_${options.messageId}_completed`,
			type: 'assistant_message_completed',
			messageId: options.messageId,
			stopReason: options.stopReason,
			usage,
			...(options.stopReason === 'error' ? { error: 'socket hang up' } : {}),
		},
	];
}

const initialRecords: ConversationRecord[] = [
	{
		...envelope,
		id: 'record_conversation_created',
		type: 'conversation_created',
		kind: 'root',
		affinityKey: 'affinity_retry',
		createdAt: timestamp,
	},
	{
		...envelope,
		id: 'record_user',
		type: 'user_message',
		messageId: 'entry_user',
		parentId: null,
		content: [{ type: 'text', text: 'Write the answer.' }],
	},
];
const failedStep = assistantRecords({
	messageId: 'entry_failed',
	parentId: 'entry_user',
	text: 'partial-A',
	stopReason: 'error',
});
const replacementStep = assistantRecords({
	messageId: 'entry_replacement',
	parentId: 'entry_failed',
	text: 'full-answer-B',
	stopReason: 'stop',
});

function reduce(records: ConversationRecord[]) {
	return reduceConversationRecords(createReducedInstanceState(), records);
}

describe('transient retry conversation projection', () => {
	it('keeps the latest model error visible until a replacement starts', () => {
		const snapshot = projectAgentConversationSnapshot(reduce([...initialRecords, ...failedStep]));
		expect(snapshot?.messages[1]).toMatchObject({
			id: 'entry_failed',
			role: 'assistant',
			parts: [{ type: 'text', text: 'partial-A', state: 'done' }],
		});
	});

	it('removes superseded partial text from snapshots and replies', () => {
		const snapshot = projectAgentConversationSnapshot(
			reduce([...initialRecords, ...failedStep, ...replacementStep]),
		);
		expect(snapshot?.messages[1]).toMatchObject({
			id: 'entry_failed',
			role: 'assistant',
			parts: [{ type: 'text', text: 'full-answer-B', state: 'done' }],
		});
		expect(snapshot?.messages).toHaveLength(2);
		expect(snapshot && replyFromSnapshot(snapshot, 'sub_retry')).toMatchObject({
			text: 'full-answer-B',
		});
	});

	it('resets live clients when the replacement step starts', () => {
		const before = reduce([...initialRecords, ...failedStep]);
		const replacementStarted = replacementStep[0];
		expect(replacementStarted).toBeDefined();
		if (!replacementStarted) return;
		const after = reduce([...initialRecords, ...failedStep, replacementStarted]);

		expect(
			projectAgentConversationBatch({
				state: after,
				previousState: before,
				records: [replacementStarted],
				batchOrdinal: 8,
			}),
		).toEqual([
			{
				type: 'conversation-reset',
				conversationId: 'conv_retry',
				position: { batch: 8, index: 0 },
				snapshot: expect.objectContaining({
					messages: [
						expect.objectContaining({ role: 'user' }),
						expect.objectContaining({
							id: 'entry_failed',
							role: 'assistant',
							parts: [],
						}),
					],
				}),
			},
		]);
	});
});
