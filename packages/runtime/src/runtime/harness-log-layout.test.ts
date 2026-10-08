import { describe, expect, it } from 'vitest';
import type { AgentSubmissionStore } from '../agent-execution-store.ts';
import { sqlite } from '../node/agent-execution-store.ts';
import { InMemoryConversationStreamStore } from './conversation-stream-store.ts';
import { createFlueLogStore } from './harness-log-store.ts';

const identity = { agentName: 'test-agent', instanceId: 'instance-1' };
const path = 'agents/test-agent/instance-1';

async function openSqliteStores() {
	const adapter = sqlite();
	await adapter.migrate?.();
	const stores = await adapter.connect();
	return {
		submissions: stores.submissionStore,
		streams: stores.conversationStreamStore,
	};
}

/** A Flue `state_write` record of the conversation `conversation-1`. */
function stateWrite(id: string, value: number) {
	return {
		v: 1 as const,
		id,
		type: 'state_write' as const,
		conversationId: 'conversation-1',
		harness: 'default',
		session: 'default',
		timestamp: '2026-10-08T00:00:00.000Z',
		name: 'count',
		value,
	};
}

/** A Flue record of `submission-1` that `attemptId` wrote. */
function ownedStateWrite(id: string, attemptId: string) {
	return { ...stateWrite(id, 1), submissionId: 'submission-1', attemptId };
}

const authorization = (attemptId: string) => ({ submissionId: 'submission-1', attemptId });

async function claimRunningSubmission(submissions: AgentSubmissionStore) {
	await submissions.admitDirect({
		kind: 'direct',
		submissionId: 'submission-1',
		agent: 'test-agent',
		id: 'instance-1',
		message: { kind: 'user', body: 'Hello' },
		acceptedAt: '2026-10-08T00:00:00.000Z',
	});
	await submissions.markSubmissionCanonicalReady('submission-1');
	await submissions.claimSubmission({
		submissionId: 'submission-1',
		attemptId: 'attempt-1',
		ownerId: 'host-a',
		leaseExpiresAt: Date.now() + 60_000,
	});
}

describe('a harness log append in the stream', () => {
	it('keeps the Flue records at the top level, first, and the harness record last', async () => {
		const streams = new InMemoryConversationStreamStore();
		const log = createFlueLogStore(streams, { producerId: 'host-a', identityFor: () => identity });
		const input = { type: 'harness.input', inputId: 'input-1', at: 1 };
		const event = {
			type: 'harness.event',
			operationId: 'operation-1',
			event: { type: 'RUN_STARTED' },
		};

		await log.append(path, 1, [
			input,
			{ ...stateWrite('record-1', 1), thread: 'conversation-1' },
			event,
			{ ...stateWrite('record-2', 2), thread: 'thread-2' },
		]);

		const [batch] = (await streams.read(path)).batches;
		expect(batch?.records.slice(0, 2)).toEqual([
			stateWrite('record-1', 1),
			stateWrite('record-2', 2),
		]);
		expect(batch?.records[2]).toMatchObject({
			type: 'harness_log_batch',
			seq: 1,
			records: [
				input,
				{ type: 'harness.flue_record', index: 0, thread: 'conversation-1' },
				event,
				{ type: 'harness.flue_record', index: 1, thread: 'thread-2' },
			],
		});
		expect(batch?.records).toHaveLength(3);
		expect(await log.read(path)).toEqual([
			{ seq: 1, record: input },
			{ seq: 2, record: { ...stateWrite('record-1', 1), thread: 'conversation-1' } },
			{ seq: 3, record: event },
			{ seq: 4, record: { ...stateWrite('record-2', 2), thread: 'thread-2' } },
		]);
	});

	it('writes an append of only Flue records plain, one position per record', async () => {
		const streams = new InMemoryConversationStreamStore();
		const log = createFlueLogStore(streams, { producerId: 'host-a', identityFor: () => identity });

		await log.append(path, 1, [{ type: 'host.note', text: 'first' }]);
		await log.append(path, 2, [
			{ ...stateWrite('record-1', 1), thread: 'conversation-1' },
			{ ...stateWrite('record-2', 2), thread: 'conversation-1' },
		]);
		await log.append(path, 4, [{ type: 'host.note', text: 'last' }]);

		const [, plain] = (await streams.read(path)).batches;
		expect(plain?.records).toEqual([stateWrite('record-1', 1), stateWrite('record-2', 2)]);
		const reader = createFlueLogStore(streams, {
			producerId: 'host-b',
			identityFor: () => identity,
		});
		expect(await reader.read(path, { after: 1 })).toEqual([
			{ seq: 2, record: { ...stateWrite('record-1', 1), thread: 'conversation-1' } },
			{ seq: 3, record: { ...stateWrite('record-2', 2), thread: 'conversation-1' } },
			{ seq: 4, record: { type: 'host.note', text: 'last' } },
		]);
	});
});

describe('attempt ownership of a harness log append', () => {
	it('writes the Flue records of the current attempt', async () => {
		const { streams, submissions } = await openSqliteStores();
		await claimRunningSubmission(submissions);
		const log = createFlueLogStore(streams, { producerId: 'host-a', identityFor: () => identity });

		await log.append(path, 1, [
			{ type: 'host.note', text: 'before' },
			{
				...ownedStateWrite('record-1', 'attempt-1'),
				thread: 'conversation-1',
				authorization: authorization('attempt-1'),
			},
		]);

		expect((await streams.read(path)).batches[0]?.records[0]).toEqual(
			ownedStateWrite('record-1', 'attempt-1'),
		);
	});

	it('fails for an attempt that no longer owns the submission, and writes nothing', async () => {
		const { streams, submissions } = await openSqliteStores();
		await claimRunningSubmission(submissions);
		await submissions.replaceSubmissionAttempt(
			{ submissionId: 'submission-1', attemptId: 'attempt-1' },
			'attempt-2',
			{ ownerId: 'host-b', leaseExpiresAt: Date.now() + 60_000 },
		);
		const log = createFlueLogStore(streams, { producerId: 'host-a', identityFor: () => identity });

		await expect(
			log.append(path, 1, [
				{ type: 'host.note', text: 'stale' },
				{
					...ownedStateWrite('record-1', 'attempt-1'),
					thread: 'conversation-1',
					authorization: authorization('attempt-1'),
				},
			]),
		).rejects.toMatchObject({
			meta: { reason: 'Submission attempt no longer owns work for this agent instance.' },
		});
		expect((await streams.read(path)).batches).toEqual([]);
	});

	it('fails for owned Flue records without an authorization', async () => {
		const { streams, submissions } = await openSqliteStores();
		await claimRunningSubmission(submissions);
		const log = createFlueLogStore(streams, { producerId: 'host-a', identityFor: () => identity });

		await expect(
			log.append(path, 1, [
				{ ...ownedStateWrite('record-1', 'attempt-1'), thread: 'conversation-1' },
			]),
		).rejects.toMatchObject({
			meta: { reason: 'Submission-owned records require an attempt authorization.' },
		});
		expect((await streams.read(path)).batches).toEqual([]);
	});

	it('fails for Flue records of two attempts in one append', async () => {
		const streams = new InMemoryConversationStreamStore();
		const log = createFlueLogStore(streams, { producerId: 'host-a', identityFor: () => identity });

		await expect(
			log.append(path, 1, [
				{
					...ownedStateWrite('record-1', 'attempt-1'),
					thread: 'conversation-1',
					authorization: authorization('attempt-1'),
				},
				{
					...ownedStateWrite('record-2', 'attempt-2'),
					thread: 'conversation-1',
					authorization: authorization('attempt-2'),
				},
			]),
		).rejects.toThrow('holds Flue records of two submission attempts');
		expect((await streams.read(path)).batches).toEqual([]);
	});

	it('writes a settlement in a stream batch of its own, at the next position', async () => {
		const { streams, submissions } = await openSqliteStores();
		await claimRunningSubmission(submissions);
		const log = createFlueLogStore(streams, { producerId: 'host-a', identityFor: () => identity });
		const settled = {
			v: 1 as const,
			id: 'record-settled',
			type: 'submission_settled' as const,
			conversationId: 'conversation-1',
			harness: 'default',
			session: 'default',
			timestamp: '2026-10-08T00:00:00.000Z',
			submissionId: 'submission-1',
			attemptId: 'attempt-1',
			outcome: 'completed' as const,
		};

		await log.append(path, 1, [
			{ type: 'host.note', text: 'before' },
			{ ...settled, thread: 'conversation-1', authorization: authorization('attempt-1') },
		]);

		const batches = (await streams.read(path)).batches;
		expect(batches.map((batch) => batch.records.map((record) => record.type))).toEqual([
			['harness_log_batch'],
			['submission_settled'],
		]);
		expect(batches[1]?.records).toEqual([settled]);
		expect(await log.read(path)).toEqual([
			{ seq: 1, record: { type: 'host.note', text: 'before' } },
			{ seq: 2, record: { ...settled, thread: 'conversation-1' } },
		]);
	});
});
