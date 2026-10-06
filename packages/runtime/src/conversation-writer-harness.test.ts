import { toolDefinition } from '@tanstack/ai';
import { fakeText } from '@tanstack/ai/testing';
import { createHarnessHost, defineHarness, durableTool } from '@tanstack/ai-harness';
import { defineAIPersistence } from '@tanstack/ai-persistence';
import { describe, expect, it } from 'vitest';
import type { HarnessLogRecord } from './conversation-records.ts';
import { ConversationRecordWriter } from './conversation-writer.ts';
import { InMemoryConversationStreamStore } from './runtime/conversation-stream-store.ts';
import { createFlueLogStore } from './runtime/harness-log-store.ts';
import { generateTaskId } from './runtime/ids.ts';
import { createTaskSessionName } from './session-identity.ts';

const path = 'agents/support/instance-1';
const identity = { agentName: 'support', instanceId: 'instance-1' };
const timestamp = '2026-10-06T00:00:00.000Z';
const root = {
	conversationId: 'conv_root',
	harness: 'default',
	session: 'default',
};

/** A durable harness session on the memory stream store, and a Flue writer over its log. */
async function openWriter(
	options: {
		onToolCall?: (append: (records: readonly HarnessLogRecord[]) => void) => void;
	} = {},
) {
	const store = new InMemoryConversationStreamStore();
	const fake = fakeText();
	fake.setResponses([
		{ toolCalls: [{ id: 'call-1', name: 'remember', input: {} }] },
		{ text: 'Done.' },
	]);
	const remember = durableTool(
		toolDefinition({ name: 'remember', description: 'Remember.' }),
		async (_args, { append }) => {
			options.onToolCall?.(append);
			return 'ok';
		},
	);
	const host = createHarnessHost({
		persistence: defineAIPersistence({
			stores: {
				log: createFlueLogStore(store, {
					producerId: 'host',
					identityFor: () => identity,
				}),
				leases: {
					acquire: async () => {},
					renew: async () => {},
					release: async () => {},
					isAlive: async () => false,
				},
			},
		}),
	});
	const session = await host.open(
		defineHarness({
			name: 'flue/writer-test',
			adapter: fake,
			tools: [remember],
		}),
		{
			threadId: root.conversationId,
			logId: path,
		},
	);
	const writer = await ConversationRecordWriter.overHarness({
		store,
		path,
		identity,
		append: (records) => session.append(records),
	});
	await writer.ensureConversation({
		...root,
		kind: 'root',
		affinityKey: 'affinity_root',
		createdAt: timestamp,
	});
	return { host, session, writer };
}

describe('ConversationRecordWriter.overHarness', () => {
	it('folds the records it appends through the harness log', async () => {
		const { host, writer } = await openWriter();

		await writer.append([
			{
				...root,
				v: 1,
				id: 'record_user',
				type: 'user_message',
				timestamp,
				messageId: 'entry_user',
				parentId: null,
				content: [{ type: 'text', text: 'Hello.' }],
			},
		]);

		expect([...((await writer.getConversation('conv_root'))?.entries.keys() ?? [])]).toEqual([
			'entry_user',
		]);
		await host.close();
	});

	it('lands a record staged in a tool call when the tool batch commits', async () => {
		let stageWrite: ((append: (records: readonly HarnessLogRecord[]) => void) => void) | undefined;
		const { host, session, writer } = await openWriter({
			onToolCall: (append) => stageWrite?.(append),
		});
		stageWrite = (append) =>
			void writer.append(
				[
					{
						...root,
						v: 1,
						id: 'record_state',
						type: 'state_write',
						timestamp,
						name: 'count',
						value: 3,
					},
				],
				{ stage: append },
			);

		await session.prompt('Remember it.', { inputId: 'input-1' });
		await session.settled('input-1');

		expect((await writer.loadReducedState()).state.get('count')).toBe(3);
		await host.close();
	});

	it('writes a child conversation and its parent link in one append', async () => {
		const { host, writer } = await openWriter();
		const taskId = generateTaskId();
		const session = createTaskSessionName(root.session, taskId);

		await writer.ensureChildConversation({
			parent: root,
			child: {
				kind: 'task',
				conversationId: 'conv_child',
				harness: root.harness,
				session,
				affinityKey: 'affinity_child',
				createdAt: timestamp,
				parentConversationId: root.conversationId,
				taskId,
			},
			ref: {
				type: 'task',
				conversationId: 'conv_child',
				harness: root.harness,
				session,
				taskId,
			},
		});

		const state = await writer.loadReducedState();
		expect(state.conversations.get('conv_child')?.kind).toBe('task');
		expect([...(state.conversations.get('conv_root')?.childConversations.keys() ?? [])]).toEqual([
			'conv_child',
		]);
		await host.close();
	});
});
