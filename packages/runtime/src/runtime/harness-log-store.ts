/**
 * The TanStack harness `LogStore` over Flue's `ConversationStreamStore`. The
 * log id is the agent stream path.
 *
 * Each harness append is one Flue batch. Its Flue records come first, at the
 * top level, as the direct path writes them. When the append has other
 * records (harness records), one `harness_log_batch` record comes last: it
 * holds every log record in log order, with a slot for each Flue record, and
 * the position of the first one. A batch with only Flue records (Flue's own
 * records from before the harness, too) reads as one position per record, so
 * an old stream opens as the start of the log.
 *
 * A Flue record carries the attempt that may write it as `authorization`.
 * The log store gives it to the stream store as `submission`, so the store
 * checks attempt ownership as it does for a direct append. A
 * `submission_settled` record goes in a stream batch of its own, because the
 * store accepts a settlement of a settling submission only alone.
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
	type HarnessFlueRecordSlot,
	type HarnessLogBatchRecord,
	type HarnessLogRecord,
	type HarnessRecordAuthorization,
	isFlueRecord,
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
		// One append holds the records of one attempt at most.
		authorizationOf(logId, records);
		for (const segment of segmentsOf(records)) {
			// Only a stream batch with owned Flue records names the attempt, so a
			// batch of harness records does not depend on the submission row.
			const submission = authorizationOf(logId, segment);
			const current = state.claim ?? (await claim(logId, state));
			const seq = state.nextSeq;
			const { offset } = await store.append({
				path: logId,
				producerId: current.producerId,
				producerEpoch: current.producerEpoch,
				incarnation: current.incarnation,
				producerSequence: current.nextProducerSequence,
				...(submission ? { submission } : {}),
				records: streamRecordsOf(segment, seq),
			});
			current.nextProducerSequence += 1;
			state.batches.push({ offset, seq, count: segment.length });
			state.nextSeq = seq + segment.length;
			state.readOffset = offset;
		}
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
				const records = harnessLogRecordsOf(batch.records);
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

/** The harness part of a stream batch: its last record, when it is one. */
function harnessBatchOf(records: readonly ConversationRecord[]) {
	const last = records.at(-1);
	return last?.type === 'harness_log_batch' ? last : undefined;
}

/**
 * The harness log records of one stream batch, in log order, as copies. A
 * Flue record slot becomes its Flue record with its `thread`. A batch with
 * only Flue records gives each record the thread of its conversation, as the
 * conversation writer stamps it.
 *
 * @example
 * ```ts
 * const page = await streams.read(path);
 * const appends = page.batches.map((batch) => harnessLogRecordsOf(batch.records));
 * ```
 */
export function harnessLogRecordsOf(records: readonly ConversationRecord[]) {
	const harnessBatch = harnessBatchOf(records);
	const logRecords: HarnessLogRecord[] = harnessBatch
		? harnessBatch.records.map((record) => {
				if (!isFlueRecordSlot(record)) return { ...record };
				const flueRecord = records[record.index];
				if (!flueRecord || flueRecord === harnessBatch) {
					throw new Error(`[flue] Harness log batch has no Flue record at index ${record.index}.`);
				}
				return { ...flueRecord, thread: record.thread };
			})
		: records.map((record) => ({ ...record, thread: record.conversationId }));
	return logRecords;
}

/**
 * The stream records of log records at positions `seq` and on: the Flue
 * records at the top level without `thread` and `authorization`, then one
 * `harness_log_batch` record when there are other records.
 */
function streamRecordsOf(records: readonly LogRecord[], seq: number) {
	const top: ConversationRecord[] = [];
	const logRecords = records.map((record) => {
		if (!isFlueRecord(record)) return { ...record };
		const { thread, authorization: _authorization, ...flueRecord } = record;
		const slot: HarnessFlueRecordSlot = {
			type: 'harness.flue_record',
			index: top.length,
			thread: typeof thread === 'string' ? thread : flueRecord.conversationId,
		};
		top.push(flueRecord);
		return slot;
	});
	if (top.length === records.length) return top;
	const batch: HarnessLogBatchRecord = {
		v: 1,
		id: generateConversationRecordId(),
		type: 'harness_log_batch',
		conversationId: '',
		harness: '',
		session: '',
		timestamp: new Date().toISOString(),
		seq,
		records: logRecords,
	};
	return [...top, batch];
}

/**
 * The log records in the stream batches they go in: each
 * `submission_settled` record alone, the records between them together.
 */
function segmentsOf(records: readonly LogRecord[]) {
	const segments: LogRecord[][] = [];
	let current: LogRecord[] = [];
	for (const record of records) {
		if (record.type !== 'submission_settled' || !isFlueRecord(record)) {
			current.push(record);
			continue;
		}
		if (current.length > 0) segments.push(current);
		segments.push([record]);
		current = [];
	}
	if (current.length > 0) segments.push(current);
	return segments;
}

/** The one attempt that may write the Flue records of an append, if any. */
function authorizationOf(logId: string, records: readonly LogRecord[]) {
	let found: HarnessRecordAuthorization | undefined;
	for (const record of records) {
		if (!isFlueRecord(record)) continue;
		const authorization = authorizationField(record);
		if (!authorization) continue;
		if (
			found &&
			(found.submissionId !== authorization.submissionId ||
				found.attemptId !== authorization.attemptId)
		) {
			throw new Error(
				`[flue] A harness log append to "${logId}" holds Flue records of two submission attempts.`,
			);
		}
		found = authorization;
	}
	return found;
}

function authorizationField(record: object): HarnessRecordAuthorization | undefined {
	if (!('authorization' in record)) return undefined;
	const { authorization } = record;
	if (
		typeof authorization === 'object' &&
		authorization !== null &&
		'submissionId' in authorization &&
		typeof authorization.submissionId === 'string' &&
		'attemptId' in authorization &&
		typeof authorization.attemptId === 'string'
	) {
		return { submissionId: authorization.submissionId, attemptId: authorization.attemptId };
	}
	return undefined;
}

function isFlueRecordSlot(record: LogRecord): record is HarnessFlueRecordSlot & LogRecord {
	return (
		record.type === 'harness.flue_record' &&
		typeof record.index === 'number' &&
		typeof record.thread === 'string'
	);
}

function isFenced(error: unknown) {
	return error instanceof ConversationStreamStoreError && FENCED_REASONS.has(error.meta?.reason);
}
