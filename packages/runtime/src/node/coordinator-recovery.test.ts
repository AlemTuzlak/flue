import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { LEASE_DURATION_MS, type PersistenceAdapter } from '../agent-execution-store.ts';
import type { ConversationRecord } from '../conversation-records.ts';
import { type AgentFunction, init, useModel, useTool } from '../index.ts';
import type { ConversationStreamStore } from '../runtime/conversation-stream-store.ts';
import {
	type FauxContext,
	fauxAssistantMessage,
	fauxProvider,
	fauxText,
	fauxToolCall,
} from '../test-utils/faux.ts';
import { sqlite, start } from './index.ts';

let directory: string;

beforeEach(async () => {
	directory = await mkdtemp(join(tmpdir(), 'flue-coordinator-recovery-'));
});

afterEach(async () => {
	vi.restoreAllMocks();
	await rm(directory, { recursive: true, force: true });
});

/** Move `Date.now` past the lease of every running submission. */
function expireLeases() {
	const realNow = Date.now.bind(Date);
	vi.spyOn(Date, 'now').mockImplementation(() => realNow() + LEASE_DURATION_MS + 1_000);
}

/**
 * The SQLite database file of the test. `stepped` resolves once a harness
 * append that holds a `harness.tool.step` record is durable.
 */
function stepWatchingDatabase(path: string) {
	const database = sqlite(path);
	const stepped = Promise.withResolvers<void>();
	const adapter: PersistenceAdapter = {
		migrate: () => database.migrate?.(),
		close: () => database.close?.(),
		async connect() {
			const stores = await database.connect();
			const stream = stores.conversationStreamStore;
			const watching: ConversationStreamStore = {
				createStream: (...args) => stream.createStream(...args),
				acquireProducer: (...args) => stream.acquireProducer(...args),
				async append(input) {
					const result = await stream.append(input);
					if (input.records.some(holdsToolStep)) stepped.resolve();
					return result;
				},
				read: (...args) => stream.read(...args),
				getMeta: (...args) => stream.getMeta(...args),
				subscribe: (...args) => stream.subscribe(...args),
				...(stream.putFoldCheckpoint
					? { putFoldCheckpoint: stream.putFoldCheckpoint.bind(stream) }
					: {}),
				...(stream.getFoldCheckpoint
					? { getFoldCheckpoint: stream.getFoldCheckpoint.bind(stream) }
					: {}),
			};
			return { ...stores, conversationStreamStore: watching };
		},
	};
	return { adapter, stepped: stepped.promise };
}

function holdsToolStep(record: ConversationRecord) {
	return (
		record.type === 'harness_log_batch' &&
		record.records.some((inner) => inner.type === 'harness.tool.step')
	);
}

/** A promise that rejects when `signal` aborts. */
function untilAborted(signal: AbortSignal | undefined) {
	return new Promise<never>((_, reject) => {
		signal?.addEventListener('abort', () => reject(new Error('Aborted.')), { once: true });
	});
}

/**
 * An agent with one durable tool. Its first run finishes the `charge` step,
 * and then waits until its turn stops. Later runs answer with the receipt.
 */
function chargeAgent() {
	const counts = { runs: 0, charges: 0 };
	const ChargeAgent: AgentFunction = () => {
		useModel('faux/model');
		useTool({
			name: 'charge',
			description: 'Charge the card.',
			durable: true,
			run: async ({ step, signal }) => {
				counts.runs += 1;
				const receipt = await step.do('charge', async () => {
					counts.charges += 1;
					return `receipt-${counts.charges}`;
				});
				if (counts.runs === 1) return untilAborted(signal);
				return `charged with ${receipt}`;
			},
		});
		return 'Charge the card.';
	};
	return { ChargeAgent, counts };
}

/**
 * The faux model. A request with a tool result gets the answer, `Ping.` gets
 * `Pong.`, and any other request calls the `charge` tool.
 */
function chargeModel() {
	const faux = fauxProvider({ models: [{ id: 'model' }] });
	const answer = (context: FauxContext) => {
		if (context.messages.some((message) => message.role === 'toolResult')) {
			return fauxAssistantMessage([fauxText('Charged.')]);
		}
		const last = context.messages.at(-1);
		if (last?.role === 'user' && JSON.stringify(last.content).includes('Ping.')) {
			return fauxAssistantMessage([fauxText('Pong.')]);
		}
		return fauxAssistantMessage([fauxToolCall('charge', {}, { id: 'call_charge' })], {
			stopReason: 'toolUse',
		});
	};
	faux.setResponses([answer, answer, answer, answer]);
	return faux;
}

it('recovers a turn that a graceful shutdown stopped during a durable tool, on the next start', async () => {
	const { ChargeAgent, counts } = chargeAgent();
	const faux = chargeModel();
	const file = join(directory, 'flue.db');

	const first = stepWatchingDatabase(file);
	const runtimeA = await start({
		agents: [ChargeAgent],
		db: first.adapter,
		providers: [faux.provider],
		env: {},
	});
	const receipt = await init(ChargeAgent, { id: 'card-1' }).dispatch('Charge it.');
	await first.stepped;
	await runtimeA.stop();
	expect(counts).toEqual({ runs: 1, charges: 1 });

	expireLeases();
	const runtimeB = await start({
		agents: [ChargeAgent],
		db: stepWatchingDatabase(file).adapter,
		providers: [faux.provider],
		env: {},
	});
	try {
		await expect(init(ChargeAgent, { id: 'card-1' }).read(receipt)).resolves.toMatchObject({
			text: 'Charged.',
		});
		expect(counts).toEqual({ runs: 2, charges: 1 });
	} finally {
		await runtimeB.stop();
	}
	const database = sqlite(file);
	const stores = await database.connect();
	await expect(stores.submissionStore.getSubmission(receipt.submissionId)).resolves.toMatchObject({
		status: 'settled',
		attemptCount: 2,
	});
	await database.close?.();
});

it('recovers a running submission whose owner is gone when the lease scan finds it', async () => {
	const { ChargeAgent, counts } = chargeAgent();
	const faux = chargeModel();
	const file = join(directory, 'flue.db');

	const first = stepWatchingDatabase(file);
	const runtimeA = await start({
		agents: [ChargeAgent],
		db: first.adapter,
		providers: [faux.provider],
		env: {},
	});
	const receipt = await init(ChargeAgent, { id: 'card-2' }).dispatch('Charge it.');
	await first.stepped;
	await runtimeA.stop();

	// The lease of the stopped owner is still valid: the startup pass leaves it.
	const runtimeB = await start({
		agents: [ChargeAgent],
		db: stepWatchingDatabase(file).adapter,
		providers: [faux.provider],
		env: {},
	});
	const database = sqlite(file);
	const stores = await database.connect();
	try {
		await expect(stores.submissionStore.getSubmission(receipt.submissionId)).resolves.toMatchObject(
			{ status: 'running', attemptCount: 1 },
		);
		expect(counts).toEqual({ runs: 1, charges: 1 });

		// The lease expires. A dispatch to another instance wakes the claim
		// loop, and its next pass runs the lease scan.
		expireLeases();
		const other = init(ChargeAgent, { id: 'card-other' });
		await expect(other.read(await other.dispatch('Ping.'))).resolves.toMatchObject({
			text: 'Pong.',
		});
		await expect(init(ChargeAgent, { id: 'card-2' }).read(receipt)).resolves.toMatchObject({
			text: 'Charged.',
		});
		expect(counts).toEqual({ runs: 2, charges: 1 });
	} finally {
		await runtimeB.stop();
	}
	await expect(stores.submissionStore.getSubmission(receipt.submissionId)).resolves.toMatchObject({
		status: 'settled',
		attemptCount: 2,
	});
	await database.close?.();
});
