import { fakeText } from '@tanstack/ai/testing';
import { defineHarness, logMessageStore } from '@tanstack/ai-harness';
import { describe, expect, it } from 'vitest';
import type { AgentSubmissionStore } from '../agent-execution-store.ts';
import { sqlite } from '../node/agent-execution-store.ts';
import type { ConversationStreamStore } from './conversation-stream-store.ts';
import { createFlueLeaseStore } from './harness-lease-store.ts';
import { createFlueLogStore } from './harness-log-store.ts';
import { createInstanceHarnessHost } from './instance-harness-host.ts';

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

/** A host on new SQLite stores, and a harness whose model answers once. */
async function openInstanceHost(answer: string) {
	const { streams, submissions } = await openSqliteStores();
	const host = createInstanceHarnessHost({
		streams,
		submissions,
		path,
		identity,
		ownerId: 'host-a',
	});
	const fake = fakeText();
	fake.setResponses([{ text: answer }]);
	const harness = defineHarness({ name: 'flue/instance-host-test', adapter: fake });
	return { streams, host, harness };
}

/** The transcript of `threadId` in the instance stream, read by a new reader. */
async function storedTranscript(streams: ConversationStreamStore, threadId: string) {
	const log = createFlueLogStore(streams, {
		producerId: 'reader',
		identityFor: () => identity,
	});
	const messages = await logMessageStore({ store: log, logId: path }).loadThread(threadId);
	return messages.map(({ role, content }) => ({ role, content }));
}

describe('the instance harness host', () => {
	it('runs a turn on a thread of the instance stream', async () => {
		const { streams, host, harness } = await openInstanceHost('Hello there.');
		const session = await host.open(harness, { threadId: 'conversation-1', logId: path });

		const turn = await session.prompt('Hi', { inputId: 'submission-1' });
		await host.close();

		expect(turn.text).toBe('Hello there.');
		expect(await storedTranscript(streams, 'conversation-1')).toEqual([
			{ role: 'user', content: 'Hi' },
			{ role: 'assistant', content: 'Hello there.' },
		]);
	});

	it('fails a turn on a thread outside the instance stream, and creates no stream for it', async () => {
		const { streams, host, harness } = await openInstanceHost('Hello there.');
		const session = await host.open(harness, { threadId: 'conversation-1' });

		const turn = Promise.resolve(session.prompt('Hi', { inputId: 'submission-1' }));
		await expect(turn).rejects.toThrow('cannot write the log "conversation-1"');
		await host.close();

		expect((await streams.read('conversation-1')).batches).toEqual([]);
	});
});

async function claimRunningSubmission(submissions: AgentSubmissionStore, leaseExpiresAt: number) {
	await submissions.admitDirect({
		kind: 'direct',
		submissionId: 'submission-1',
		agent: 'test-agent',
		id: 'instance-1',
		message: { kind: 'user', body: 'Hello' },
		acceptedAt: '2026-10-05T00:00:00.000Z',
	});
	await submissions.markSubmissionCanonicalReady('submission-1');
	await submissions.claimSubmission({
		submissionId: 'submission-1',
		attemptId: 'attempt-1',
		ownerId: 'host-a',
		leaseExpiresAt,
	});
}

/** A lease view where this process runs the attempts of `runningHere`. */
function leasesRunning(submissions: AgentSubmissionStore, runningHere: readonly string[]) {
	return createFlueLeaseStore(submissions, 'host-a', {
		isLive: (inputId) => runningHere.includes(inputId),
	});
}

const leaseKey = {
	threadId: 'conversation-1',
	inputId: 'submission-1',
	operationId: 'operation-1',
	attempt: 1,
};

describe('Flue lease view with a liveness function', () => {
	it('is alive while this process runs the submission, after the row lease expired', async () => {
		const { submissions } = await openSqliteStores();
		await claimRunningSubmission(submissions, Date.now() - 1_000);

		expect(await leasesRunning(submissions, ['submission-1']).isAlive(leaseKey)).toBe(true);
	});

	it('is not alive when this process does not run the submission, while the row lease runs', async () => {
		const { submissions } = await openSqliteStores();
		await claimRunningSubmission(submissions, Date.now() + 30_000);

		expect(await leasesRunning(submissions, []).isAlive(leaseKey)).toBe(false);
	});
});
