import { DatabaseSync } from 'node:sqlite';
import { afterEach, expect, it, vi } from 'vitest';
import { createFlueContext } from '../client.ts';
import { type AgentFunction, useModel, useTool } from '../index.ts';
import { setProvider } from '../providers/registry.ts';
import { createDirectAgentSubmissionInput } from '../runtime/agent-submissions.ts';
import { readSubmissionReply } from '../runtime/conversation-observer.ts';
import { harnessLogRecordsOf } from '../runtime/harness-log-store.ts';
import { resolveModel } from '../runtime/providers.ts';
import {
	type FauxContext,
	fauxAssistantMessage,
	fauxProvider,
	fauxText,
	fauxToolCall,
} from '../test-utils/faux.ts';
import { createCloudflareAgentRuntime } from './agent-coordinator.ts';

const AGENT_NAME = 'ChargeAgent';
const INSTANCE_NAME = 'card-1';
const STREAM_PATH = `agents/${AGENT_NAME}/${INSTANCE_NAME}`;

afterEach(() => {
	vi.restoreAllMocks();
});

function queryExpectsRows(query: string) {
	const trimmed = query.trimStart().toUpperCase();
	return trimmed.startsWith('SELECT') || trimmed.startsWith('WITH') || /\bRETURNING\b/i.test(query);
}

/**
 * Fake Durable Object storage on `node:sqlite`. It outlives the isolates
 * that use it, as Durable Object storage does.
 */
function fakeStorage() {
	const db = new DatabaseSync(':memory:');
	return {
		sql: {
			exec(query: string, ...bindings: Array<string | number | bigint | null | Uint8Array>) {
				const statement = db.prepare(query);
				const rows: Array<Record<string, unknown>> = [];
				if (queryExpectsRows(query)) {
					for (const row of statement.all(...bindings)) rows.push({ ...row });
				} else {
					statement.run(...bindings);
				}
				return {
					toArray: () => rows,
					one: () => {
						const row = rows[0];
						if (!row) throw new Error('Expected exactly one row.');
						return row;
					},
					raw: () => rows.map((row) => Object.values(row))[Symbol.iterator](),
					[Symbol.iterator]: () => rows[Symbol.iterator](),
					columnNames: Object.keys(rows[0] ?? {}),
					rowsRead: rows.length,
					rowsWritten: 0,
				};
			},
		},
		transactionSync<T>(closure: () => T) {
			db.exec('BEGIN');
			try {
				const result = closure();
				db.exec('COMMIT');
				return result;
			} catch (error) {
				db.exec('ROLLBACK');
				throw error;
			}
		},
	};
}

type FakeStorage = ReturnType<typeof fakeStorage>;

/**
 * One isolate of the Durable Object on `storage`: a fresh coordinator, as
 * after an isolate death. `schedule` records each wake, and `runFiber`
 * starts the fiber at once and leaves it running.
 */
function isolate(storage: FakeStorage, agent: AgentFunction) {
	const wakes: number[] = [];
	const runtime = createCloudflareAgentRuntime({
		agents: [{ name: AGENT_NAME, agent }],
		createContext: ({ instance, agentName, request, submissionId }) =>
			createFlueContext({
				id: instance.name,
				agentName,
				submissionId,
				env: {},
				req: request,
				agentConfig: { resolveModel },
			}),
		runWithInstanceContext: (_instance, _agentName, callback) => callback(),
	});
	const instance = {
		name: INSTANCE_NAME,
		env: {},
		ctx: { id: { toString: () => 'do-1' }, storage },
		async schedule(delaySeconds: number) {
			wakes.push(delaySeconds);
		},
		async runFiber(
			_name: string,
			callback: (ctx: { stash(snapshot: unknown): void }) => Promise<void>,
		) {
			await callback({ stash: () => {} });
		},
	};
	const prepared = runtime.prepare({
		storage,
		className: 'FlueChargeAgent',
		agentName: AGENT_NAME,
	});
	runtime.attach(instance, prepared);
	return {
		runtime,
		instance,
		wakes,
		submissions: prepared.submissionStore,
		streams: prepared.conversationStreamStore,
		/** The wake pass, as the alarm runs it. */
		wake: () => runtime.drainSubmissions(instance),
	};
}

type Isolate = ReturnType<typeof isolate>;

/**
 * An agent with one durable tool. The first run finishes the `charge` step
 * and then waits for `release`. Later runs answer with the receipt.
 */
function chargeAgent() {
	const counts = { runs: 0, charges: 0 };
	const charged = Promise.withResolvers<void>();
	const release = Promise.withResolvers<void>();
	const ChargeAgent: AgentFunction = () => {
		useModel('faux/model');
		useTool({
			name: 'charge',
			description: 'Charge the card.',
			durable: true,
			run: async ({ step }) => {
				counts.runs += 1;
				const receipt = await step.do('charge', async () => {
					counts.charges += 1;
					return `receipt-${counts.charges}`;
				});
				if (counts.runs === 1) {
					charged.resolve();
					await release.promise;
				}
				return `charged with ${receipt}`;
			},
		});
		return 'Charge the card.';
	};
	return { ChargeAgent, counts, charged: charged.promise, release: () => release.resolve() };
}

/**
 * The faux model. A request with a tool result gets `Charged.`, and any
 * other request calls the `charge` tool. `requests` counts the model calls.
 */
function useChargeModel() {
	const faux = fauxProvider({ models: [{ id: 'model' }] });
	const toolResults: string[] = [];
	let requests = 0;
	const answer = (context: FauxContext) => {
		requests += 1;
		for (const message of context.messages) {
			if (message.role !== 'toolResult') continue;
			for (const block of message.content) {
				if (block.type === 'text') toolResults.push(block.text);
			}
		}
		if (context.messages.some((message) => message.role === 'toolResult')) {
			return fauxAssistantMessage([fauxText('Charged.')]);
		}
		return fauxAssistantMessage([fauxToolCall('charge', {}, { id: 'call_charge' })], {
			stopReason: 'toolUse',
		});
	};
	faux.setResponses([answer, answer, answer, answer]);
	setProvider(faux.provider);
	return { toolResults, requests: () => requests };
}

/** Admit a direct submission with the wake pass, as an admission does. */
async function admit(host: Isolate) {
	const input = await createDirectAgentSubmissionInput({
		agent: AGENT_NAME,
		id: INSTANCE_NAME,
		message: { kind: 'user', body: 'Charge it.' },
	});
	await host.submissions.admitDirect(input);
	await host.wake();
	return input.submissionId;
}

async function settledSubmission(host: Isolate, submissionId: string) {
	return vi.waitFor(
		async () => {
			const submission = await host.submissions.getSubmission(submissionId);
			if (submission?.status !== 'settled') throw new Error('The submission is not settled yet.');
			return submission;
		},
		{ timeout: 10_000, interval: 20 },
	);
}

/** Every harness log record of the instance stream, in order. */
async function logRecords(host: Isolate) {
	const page = await host.streams.read(STREAM_PATH);
	return page.batches.flatMap((batch) => harnessLogRecordsOf(batch.records));
}

/** The outcome of each `submission_settled` record of `submissionId`. */
async function settledOutcomes(host: Isolate, submissionId: string) {
	return (await logRecords(host))
		.filter(
			(record) => record.type === 'submission_settled' && record.submissionId === submissionId,
		)
		.map((record) => record.outcome);
}

/** The outcome of each `harness.input.settled` record of `submissionId`. */
async function harnessSettlements(host: Isolate, submissionId: string) {
	return (await logRecords(host))
		.filter((record) => record.type === 'harness.input.settled' && record.inputId === submissionId)
		.map((record) => record.outcome);
}

/** The reply of `submissionId`, as a `read()` gets it. */
async function replyText(host: Isolate, submissionId: string) {
	const reply = await readSubmissionReply({ store: host.streams, path: STREAM_PATH, submissionId });
	return reply.text;
}

it('recovers a turn cut by an isolate death during a durable tool, through onFiberRecovered', async () => {
	const { ChargeAgent, counts, charged } = chargeAgent();
	const model = useChargeModel();
	const storage = fakeStorage();

	const first = isolate(storage, ChargeAgent);
	const submissionId = await admit(first);
	await charged;
	// The isolate dies here: its fiber never runs again.

	const second = isolate(storage, ChargeAgent);
	await second.runtime.onFiberRecovered(
		second.instance,
		{ name: 'flue:submission-attempt', snapshot: { submissionId } },
		() => {},
	);
	expect(second.wakes).toEqual([0]);
	await second.wake();

	await expect(settledSubmission(second, submissionId)).resolves.toMatchObject({
		attemptCount: 2,
	});
	expect(await settledOutcomes(second, submissionId)).toEqual(['completed']);
	expect(await harnessSettlements(second, submissionId)).toEqual(['completed']);
	expect(counts).toEqual({ runs: 2, charges: 1 });
	expect(model.toolResults).toEqual(['"charged with receipt-1"']);
	expect(await replyText(second, submissionId)).toBe('Charged.');
});

it('recovers a running submission with no active controller on the alarm wake pass', async () => {
	const { ChargeAgent, counts, charged } = chargeAgent();
	useChargeModel();
	const storage = fakeStorage();

	const first = isolate(storage, ChargeAgent);
	const submissionId = await admit(first);
	await charged;

	// A fresh isolate whose first entry is the heartbeat alarm.
	const second = isolate(storage, ChargeAgent);
	await expect(second.submissions.getSubmission(submissionId)).resolves.toMatchObject({
		status: 'running',
		attemptCount: 1,
	});
	await second.wake();

	await expect(settledSubmission(second, submissionId)).resolves.toMatchObject({
		attemptCount: 2,
	});
	expect(await settledOutcomes(second, submissionId)).toEqual(['completed']);
	expect(await harnessSettlements(second, submissionId)).toEqual(['completed']);
	expect(await replyText(second, submissionId)).toBe('Charged.');
	expect(counts).toEqual({ runs: 2, charges: 1 });
});

it('leaves a submission whose attempt runs in this isolate to that attempt on the wake pass', async () => {
	const { ChargeAgent, counts, charged, release } = chargeAgent();
	const model = useChargeModel();
	const storage = fakeStorage();

	const host = isolate(storage, ChargeAgent);
	const submissionId = await admit(host);
	await charged;
	await host.wake();
	await host.wake();

	await expect(host.submissions.getSubmission(submissionId)).resolves.toMatchObject({
		status: 'running',
		attemptCount: 1,
	});
	expect(counts).toEqual({ runs: 1, charges: 1 });
	expect(model.requests()).toBe(1);

	release();
	await expect(settledSubmission(host, submissionId)).resolves.toMatchObject({
		attemptCount: 1,
	});
	expect(await settledOutcomes(host, submissionId)).toEqual(['completed']);
	expect(await harnessSettlements(host, submissionId)).toEqual(['completed']);
	expect(await replyText(host, submissionId)).toBe('Charged.');
	expect(counts).toEqual({ runs: 1, charges: 1 });
	expect(model.requests()).toBe(2);
});
