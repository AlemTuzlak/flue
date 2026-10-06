/**
 * The TanStack harness `LogStore` over Flue's `ConversationStreamStore`. The
 * log id is the agent stream path.
 *
 * Each harness append is one Flue batch: one `harness_log_batch` record that
 * holds the records and the position of the first one. A batch from before the
 * harness (Flue's own records) reads as one position per record, so an old
 * stream opens as the start of the log.
 *
 * The log has one writer: the view holds a producer claim, and another writer
 * that claims the stream makes the next append of this view fail. The view
 * then claims again, reads the new batches, and appends only when `seq` is
 * still the next free position. Otherwise the append fails with
 * `LogConflictError`, as the `LogStore` contract asks.
 */
import {
	LogConflictError,
	type LogEntry,
	type LogRecord,
	type LogStore,
} from '@tanstack/ai-persistence';
import {
	type ConversationRecord,
	generateConversationRecordId,
	type HarnessLogBatchRecord,
} from '../conversation-records.ts';
import { ConversationStreamStoreError } from '../errors.ts';
import type {
	ConversationProducerClaim,
	ConversationStreamIdentity,
	ConversationStreamStore,
} from './conversation-stream-store.ts';

/** Where a Flue batch sits in the log. */
interface BatchPosition {
	offset: string;
	/** The position of the batch's first record. */
	seq: number;
	count: number;
}

interface LogState {
	/** Every batch read so far, in offset order. */
	batches: BatchPosition[];
	/** The offset of the last batch read. `-1` before the first one. */
	readOffset: string;
	/** The next free position. */
	nextSeq: number;
	claim: ConversationProducerClaim | undefined;
	/** The operations on this log run one at a time. */
	queue: Promise<unknown>;
}

/** Store reasons that mean another writer claimed the stream. */
const FENCED_REASONS: ReadonlySet<unknown> = new Set([
	'Producer ownership is stale.',
	'Producer sequence is not the next expected value.',
]);

/**
 * A `LogStore` whose logs are Flue conversation streams.
 *
 * `producerId` names this writer. `identityFor` gives the stream identity of
 * a log id, for the first append of a new stream.
 */
export function createFlueLogStore(
	store: ConversationStreamStore,
	options: {
		producerId: string;
		identityFor(logId: string): ConversationStreamIdentity;
	},
) {
	const logs = new Map<string, LogState>();

	function stateOf(logId: string) {
		const existing = logs.get(logId);
		if (existing) return existing;
		const state: LogState = {
			batches: [],
			readOffset: '-1',
			nextSeq: 1,
			claim: undefined,
			queue: Promise.resolve(),
		};
		logs.set(logId, state);
		return state;
	}

	function serially<T>(state: LogState, operation: () => Promise<T>) {
		const result = state.queue.then(operation);
		state.queue = result.catch(() => undefined);
		return result;
	}

	/** Read the batches that other writers appended since the last read. */
	async function catchUp(logId: string, state: LogState) {
		for (;;) {
			const page = await store.read(logId, { offset: state.readOffset });
			for (const batch of page.batches) {
				const harnessBatch = harnessBatchOf(batch.records);
				const seq = harnessBatch?.seq ?? state.nextSeq;
				if (seq !== state.nextSeq) {
					throw new Error(
						`[flue] Harness log "${logId}" expected position ${state.nextSeq} at offset ${batch.offset}, found ${seq}.`,
					);
				}
				const count = harnessBatch?.records.length ?? batch.records.length;
				state.batches.push({ offset: batch.offset, seq, count });
				state.nextSeq = seq + count;
				state.readOffset = batch.offset;
			}
			if (page.upToDate || page.batches.length === 0) return;
		}
	}

	async function claim(logId: string, state: LogState) {
		const claimed = await store.acquireProducer(logId, options.producerId);
		state.claim = claimed;
		return claimed;
	}

	async function appendBatch(logId: string, state: LogState, records: readonly LogRecord[]) {
		const current = state.claim ?? (await claim(logId, state));
		const batch: HarnessLogBatchRecord = {
			v: 1,
			id: generateConversationRecordId(),
			type: 'harness_log_batch',
			conversationId: '',
			harness: '',
			session: '',
			timestamp: new Date().toISOString(),
			seq: state.nextSeq,
			records: records.map((record) => ({ ...record })),
		};
		const { offset } = await store.append({
			path: logId,
			producerId: current.producerId,
			producerEpoch: current.producerEpoch,
			incarnation: current.incarnation,
			producerSequence: current.nextProducerSequence,
			records: [batch],
		});
		current.nextProducerSequence += 1;
		state.batches.push({ offset, seq: batch.seq, count: records.length });
		state.nextSeq = batch.seq + records.length;
		state.readOffset = offset;
	}

	return {
		append: (logId, seq, records) => {
			if (records.length === 0) return Promise.resolve();
			const state = stateOf(logId);
			return serially(state, async () => {
				if (!state.claim) {
					await store.createStream(logId, options.identityFor(logId));
					await claim(logId, state);
				}
				await catchUp(logId, state);
				if (seq !== state.nextSeq) throw new LogConflictError(logId, seq);
				try {
					await appendBatch(logId, state, records);
				} catch (error) {
					if (!isFenced(error)) throw error;
					// Another writer claimed the stream. Take it back, and append only
					// when nobody wrote at `seq` in between.
					await claim(logId, state);
					await catchUp(logId, state);
					if (seq !== state.nextSeq) throw new LogConflictError(logId, seq);
					await appendBatch(logId, state, records);
				}
			});
		},
		read: (logId, readOptions = {}) => {
			const { after = 0, limit } = readOptions;
			const state = stateOf(logId);
			return serially(state, async () => {
				await catchUp(logId, state);
				if (limit === 0) return [];
				return readEntries(logId, state, after, limit);
			});
		},
		subscribe: (logId, listener) => store.subscribe(logId, listener),
	} satisfies LogStore;

	async function readEntries(
		logId: string,
		state: LogState,
		after: number,
		limit: number | undefined,
	) {
		const first = state.batches.findIndex((batch) => batch.seq + batch.count - 1 > after);
		if (first === -1) return [];
		const entries: LogEntry[] = [];
		let index = first;
		let offset = state.batches[first - 1]?.offset ?? '-1';
		while (index < state.batches.length) {
			const page = await store.read(logId, { offset });
			if (page.batches.length === 0) break;
			for (const batch of page.batches) {
				const position = state.batches[index];
				if (!position) break;
				const records = recordsOf(batch.records);
				for (const [i, record] of records.entries()) {
					const seq = position.seq + i;
					if (seq <= after) continue;
					entries.push({ seq, record });
					if (entries.length === limit) return entries;
				}
				index += 1;
				offset = batch.offset;
			}
		}
		return entries;
	}
}

function harnessBatchOf(records: readonly ConversationRecord[]) {
	const [only] = records;
	return records.length === 1 && only?.type === 'harness_log_batch' ? only : undefined;
}

/** The log records of a Flue batch, as copies. */
function recordsOf(records: readonly ConversationRecord[]) {
	const harnessBatch = harnessBatchOf(records);
	if (harnessBatch) return harnessBatch.records.map((record) => ({ ...record }));
	return records.map((record) => ({ ...record }));
}

function isFenced(error: unknown) {
	return error instanceof ConversationStreamStoreError && FENCED_REASONS.has(error.meta?.reason);
}
