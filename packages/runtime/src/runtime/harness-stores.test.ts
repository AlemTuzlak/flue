import { type ModelMessage, toolDefinition } from '@tanstack/ai';
import { fakeText } from '@tanstack/ai/testing';
import { createHarnessHost, defineHarness, durableTool } from '@tanstack/ai-harness';
import { defineAIPersistence } from '@tanstack/ai-persistence';
import { runPersistenceConformance } from '@tanstack/ai-persistence/testkit';
import { describe, expect, it } from 'vitest';
import type { AgentSubmissionStore } from '../agent-execution-store.ts';
import { sqlite } from '../node/agent-execution-store.ts';
import {
	type ConversationStreamStore,
	InMemoryConversationStreamStore,
} from './conversation-stream-store.ts';
import { createFlueLeaseStore } from './harness-lease-store.ts';
import { createFlueLogStore } from './harness-log-store.ts';

const identity = { agentName: 'test-agent', instanceId: 'instance-1' };

async function openSqliteStores() {
	const adapter = sqlite();
	await adapter.migrate?.();
	const stores = await adapter.connect();
	return {
		submissions: stores.submissionStore,
		streams: stores.conversationStreamStore,
	};
}

function logPersistence(streams: ConversationStreamStore) {
	return defineAIPersistence({
		stores: {
			log: createFlueLogStore(streams, {
				producerId: 'producer',
				identityFor: () => identity,
			}),
		},
	});
}

const onlyTheLog = {
	skip: [
		'messages',
		'runs',
		'interrupts',
		'metadata',
		'generationRuns',
		'artifacts',
		'blobs',
		'inbox',
		'credentials',
		'leases',
		'activities',
	],
} satisfies Parameters<typeof runPersistenceConformance>[2];

runPersistenceConformance(
	'Flue log view on the memory stream store',
	() => logPersistence(new InMemoryConversationStreamStore()),
	onlyTheLog,
);
runPersistenceConformance(
	'Flue log view on the SQLite stream store',
	async () => logPersistence((await openSqliteStores()).streams),
	onlyTheLog,
);

describe('Flue log view', () => {
	it('reads a stream from before the harness as the start of the log', async () => {
		const streams = new InMemoryConversationStreamStore();
		await streams.createStream('agents/legacy', identity);
		const claim = await streams.acquireProducer('agents/legacy', 'old-runtime');
		await streams.append({
			path: 'agents/legacy',
			producerId: claim.producerId,
			producerEpoch: claim.producerEpoch,
			incarnation: claim.incarnation,
			producerSequence: 0,
			records: [
				{
					v: 1,
					id: 'record-1',
					type: 'state_write',
					conversationId: 'conversation-1',
					harness: 'default',
					session: 'default',
					timestamp: '2026-10-05T00:00:00.000Z',
					name: 'count',
					value: 1,
				},
			],
		});
		const log = createFlueLogStore(streams, {
			producerId: 'new-runtime',
			identityFor: () => identity,
		});

		await log.append('agents/legacy', 2, [{ type: 'host.note', text: 'after' }]);

		expect((await log.read('agents/legacy')).map(({ seq, record }) => [seq, record.type])).toEqual([
			[1, 'state_write'],
			[2, 'host.note'],
		]);
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

const leaseKey = {
	threadId: 'agents/test-agent',
	inputId: 'submission-1',
	operationId: 'operation-1',
	attempt: 1,
};

describe('Flue lease view', () => {
	it('renews the submission lease when the harness takes and renews it', async () => {
		const { submissions } = await openSqliteStores();
		const shortLease = Date.now() + 1_000;
		await claimRunningSubmission(submissions, shortLease);
		const leases = createFlueLeaseStore(submissions, 'host-a');

		await leases.acquire({
			...leaseKey,
			ownerId: 'host-a',
			expiresAt: shortLease,
		});

		const renewed = await submissions.getSubmission('submission-1');
		expect(renewed?.leaseExpiresAt).toBeGreaterThan(shortLease + 20_000);
	});

	it('is alive while the lease runs, and not after it expires', async () => {
		const live = (await openSqliteStores()).submissions;
		await claimRunningSubmission(live, Date.now() + 30_000);
		const expired = (await openSqliteStores()).submissions;
		await claimRunningSubmission(expired, Date.now() - 1_000);

		expect(await createFlueLeaseStore(live, 'host-b').isAlive(leaseKey)).toBe(true);
		expect(await createFlueLeaseStore(expired, 'host-b').isAlive(leaseKey)).toBe(false);
	});

	it('is not alive for an input that is not a submission', async () => {
		const { submissions } = await openSqliteStores();

		expect(await createFlueLeaseStore(submissions, 'host-b').isAlive(leaseKey)).toBe(false);
	});
});

/** A promise and its resolve function. */
function deferred() {
	let resolve = () => {};
	const promise = new Promise<void>((done) => {
		resolve = done;
	});
	return { promise, resolve };
}

/** Waits until `signal` aborts: a tool call that a crash cuts. */
function hang(signal: AbortSignal | undefined) {
	return new Promise<never>((_, reject) =>
		signal?.addEventListener('abort', () => reject(signal.reason)),
	);
}

/**
 * Host A runs a turn on the SQLite stores until a tool call starts, then
 * stops there. Host B opens the same thread on the same stores and recovers
 * the turn. The model's second answer records the request it got.
 */
async function crashDuringToolCall(
	tool: Parameters<typeof defineHarness>[0]['tools'],
	started: Promise<void>,
) {
	const { streams, submissions } = await openSqliteStores();
	const requests: ModelMessage[][] = [];
	const fake = fakeText();
	fake.setResponses([
		{ toolCalls: [{ id: 'call-1', name: 'work', input: {} }] },
		({ request }) => {
			requests.push(request.messages);
			return { text: 'Done.' };
		},
	]);
	const harness = defineHarness({
		name: 'flue/harness-stores-test',
		adapter: fake,
		tools: tool,
	});
	const hostOn = (ownerId: string) =>
		createHarnessHost({
			persistence: defineAIPersistence({
				stores: {
					log: createFlueLogStore(streams, {
						producerId: ownerId,
						identityFor: () => identity,
					}),
					leases: createFlueLeaseStore(submissions, ownerId),
				},
			}),
		});

	const hostA = hostOn('host-a');
	const sessionA = await hostA.open(harness, { threadId: 'agents/test-agent' });
	// Host A never finishes this turn: the tool call hangs until the host closes.
	Promise.resolve(sessionA.prompt('Do the work.', { inputId: 'input-1' })).catch(() => undefined);
	await started;

	const hostB = hostOn('host-b');
	const sessionB = await hostB.open(harness, { threadId: 'agents/test-agent' });
	const settlement = await sessionB.settled('input-1');
	await hostB.close();
	await hostA.close().catch(() => undefined);
	return {
		settlement,
		toolMessage: requests[0]?.find((message) => message.role === 'tool'),
	};
}

describe('a harness session on the Flue stores, cut during a tool call', () => {
	it("gives the model an error for a replay: 'never' tool and does not run it again", async () => {
		const started = deferred();
		let runs = 0;
		const work = durableTool(
			toolDefinition({ name: 'work', description: 'Send a notice.' }),
			async (_args, { abortSignal }) => {
				runs += 1;
				started.resolve();
				return hang(abortSignal);
			},
			{ replay: 'never' },
		);

		const { settlement, toolMessage } = await crashDuringToolCall([work], started.promise);

		expect(settlement.outcome).toBe('completed');
		expect(runs).toBe(1);
		expect(toolMessage?.error).toBeDefined();
	});

	it('runs a durable tool again, and its finished step returns the stored value', async () => {
		const started = deferred();
		let executions = 0;
		let charges = 0;
		const work = durableTool(
			toolDefinition({ name: 'work', description: 'Charge the order.' }),
			async (_args, { step, abortSignal }) => {
				executions += 1;
				const chargeId = await step.do('charge', () => {
					charges += 1;
					return 'charge-1';
				});
				if (executions === 1) {
					started.resolve();
					return hang(abortSignal);
				}
				return { chargeId };
			},
		);

		const { settlement, toolMessage } = await crashDuringToolCall([work], started.promise);

		expect(settlement.outcome).toBe('completed');
		expect({ executions, charges }).toEqual({ executions: 2, charges: 1 });
		expect(toolMessage?.content).toBe('{"chargeId":"charge-1"}');
	});
});
