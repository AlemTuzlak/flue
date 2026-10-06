import { FOLD_CHECKPOINT_INTERVAL, writeFoldCheckpoint } from './conversation-fold-checkpoint.ts';
import { type ConversationFoldHost, getConversationFoldHost } from './conversation-fold-host.ts';
import type {
	CanonicalChildSessionRef,
	ConversationCreatedRecord,
	ConversationRecord,
	HarnessLogRecord,
} from './conversation-records.ts';
import type { IndexedConversationRecord, ReducedInstanceState } from './conversation-reducer.ts';
import { conversationScopeKey, reduceConversationRecords } from './conversation-reducer.ts';
import type {
	ConversationProducerClaim,
	ConversationStreamIdentity,
	ConversationStreamStore,
} from './runtime/conversation-stream-store.ts';

export interface ConversationRecordScope {
	conversationId: string;
	harness: string;
	session: string;
}

export interface ConversationAppendOptions {
	submission?: { submissionId: string; attemptId: string };
	/**
	 * Inside a tool call on the harness: stage the records with the call's tool
	 * batch instead of appending them now. They land when the batch commits.
	 */
	stage?: (records: readonly HarnessLogRecord[]) => void;
}

/**
 * Where a writer appends. `store`: straight to the stream, behind the
 * writer's own producer claim. `harness`: through the harness log, whose log
 * store holds the claim; reads then fold the stream itself.
 */
type WriterTarget =
	| { kind: 'store'; claim: ConversationProducerClaim }
	| {
			kind: 'harness';
			append: (records: readonly HarnessLogRecord[]) => Promise<void>;
			incarnation: string;
	  };

type ConversationCreationInput = ConversationCreatedRecord extends infer Record
	? Record extends ConversationCreatedRecord
		? Omit<Record, 'v' | 'id' | 'type' | 'timestamp'>
		: never
	: never;

type WriterLifecycle = { status: 'active' } | { status: 'failed'; error: unknown };

/**
 * How long streamed deltas are coalesced before being appended to the durable
 * stream. The timer only governs mid-block streaming cadence — block boundaries
 * and message completion flush immediately. Lower = smoother live streaming
 * (deltas reach observers sooner, in smaller batches) at the cost of more
 * durable writes; higher = fewer writes but burstier streaming.
 */
const CANONICAL_FLUSH_DELAY_MS = 1000;

export class ConversationRecordWriter {
	private lifecycle: WriterLifecycle = { status: 'active' };
	private tail: Promise<void> = Promise.resolve();
	private nextProducerSequence: number;
	private reducedState: ReducedInstanceState | undefined;
	private pendingRecords: ConversationRecord[] = [];
	private pendingOptions: ConversationAppendOptions | undefined;
	private pendingTimer: ReturnType<typeof setTimeout> | undefined;
	private pendingFlush: Promise<{ offset: string }> | undefined;
	private flushing: Promise<{ offset: string }> | undefined;
	private resolvePending: ((result: { offset: string }) => void) | undefined;
	private rejectPending: ((error: unknown) => void) | undefined;
	private lastFlushStartedAt = 0;

	private readonly foldHost: ConversationFoldHost;
	private batchesSinceFoldCheckpoint = 0;

	private constructor(
		private readonly store: ConversationStreamStore,
		readonly path: string,
		private readonly target: WriterTarget,
		private readonly onFailed?: (writer: ConversationRecordWriter) => void,
	) {
		this.nextProducerSequence = target.kind === 'store' ? target.claim.nextProducerSequence : 0;
		this.foldHost = getConversationFoldHost(store, path);
		this.foldHost.pin();
	}

	static async create(options: {
		store: ConversationStreamStore;
		path: string;
		identity: ConversationStreamIdentity;
		producerId: string;
		onFailed?: (writer: ConversationRecordWriter) => void;
	}): Promise<ConversationRecordWriter> {
		await options.store.createStream(options.path, options.identity);
		const claim = await options.store.acquireProducer(options.path, options.producerId);
		return new ConversationRecordWriter(
			options.store,
			options.path,
			{ kind: 'store', claim },
			options.onFailed,
		);
	}

	/**
	 * A writer whose records go through the harness log of `path`. `append`
	 * appends host records through any open session of that log. Each record
	 * names its conversation's session with `thread` (the conversation id).
	 */
	static async overHarness(options: {
		store: ConversationStreamStore;
		path: string;
		identity: ConversationStreamIdentity;
		append: (records: readonly HarnessLogRecord[]) => Promise<void>;
		onFailed?: (writer: ConversationRecordWriter) => void;
	}): Promise<ConversationRecordWriter> {
		await options.store.createStream(options.path, options.identity);
		const meta = await options.store.getMeta(options.path);
		if (!meta) throw new Error(`[flue] Conversation stream "${options.path}" does not exist.`);
		return new ConversationRecordWriter(
			options.store,
			options.path,
			{
				kind: 'harness',
				append: options.append,
				incarnation: meta.incarnation,
			},
			options.onFailed,
		);
	}

	async loadReducedState(): Promise<ReducedInstanceState> {
		this.assertActive();
		// The harness also appends to the stream (its own records, and the
		// records a tool batch staged), so the fold follows the stream head.
		if (this.target.kind === 'harness') {
			this.reducedState = await this.foldHost.getStateAtHead();
			return this.reducedState;
		}
		if (this.reducedState) return this.reducedState;
		// The shared fold host serves the same state a from-scratch load
		// produces — and reuses a fold a read already paid for. The producer
		// fence guarantees no other writer can append behind this claim, so
		// the host's head is this writer's head.
		const sequenceBefore = this.nextProducerSequence;
		const loaded = await this.foldHost.getStateAtHead();
		this.assertActive();
		// A first append racing this load would make the loaded state stale —
		// and, once memoized, every later append would fold onto it and publish
		// the gap to the shared host. The enqueue path can append before this
		// load completes — a leading-edge flush is scheduled as a microtask —
		// so re-loading keeps the host safe if one ever does.
		if (this.nextProducerSequence !== sequenceBefore) return this.loadReducedState();
		this.reducedState ??= loaded;
		return this.reducedState;
	}

	async getConversationLeaf(conversationId: string): Promise<string | null> {
		return (await this.loadReducedState()).conversations.get(conversationId)?.activeLeafId ?? null;
	}

	async hasConversationEntry(conversationId: string, entryId: string): Promise<boolean> {
		return (
			(await this.loadReducedState()).conversations.get(conversationId)?.entries.has(entryId) ??
			false
		);
	}

	async hasRecord(recordId: string): Promise<boolean> {
		return (await this.loadReducedState()).recordsById.has(recordId);
	}

	async getRecord(recordId: string): Promise<IndexedConversationRecord | undefined> {
		return (await this.loadReducedState()).recordsById.get(recordId);
	}

	async getConversation(conversationId: string) {
		return (await this.loadReducedState()).conversations.get(conversationId);
	}

	async findConversation(harness: string, session: string) {
		const state = await this.loadReducedState();
		const conversationId = state.conversationScopes.get(conversationScopeKey(harness, session));
		return conversationId ? state.conversations.get(conversationId) : undefined;
	}

	get offset(): string {
		return this.reducedState?.recordsThroughOffset ?? this.initialOffset();
	}

	private initialOffset() {
		return this.target.kind === 'store' ? this.target.claim.offset : '-1';
	}

	get failed(): boolean {
		return this.lifecycle.status === 'failed';
	}

	append(
		records: readonly ConversationRecord[],
		options: ConversationAppendOptions = {},
	): Promise<{ offset: string }> {
		try {
			this.assertActive();
			return this.appendBatch(records, options);
		} catch (error) {
			return Promise.reject(error);
		}
	}

	enqueue(
		records: readonly ConversationRecord[],
		options: ConversationAppendOptions = {},
	): Promise<{ offset: string }> {
		try {
			this.assertActive();
			if (
				this.pendingRecords.length > 0 &&
				!sameAppendOptions(this.pendingOptions ?? {}, options)
			) {
				throw new Error(
					'[flue] Canonical batch ownership changed before the pending batch flushed.',
				);
			}
			this.pendingOptions = options;
			this.pendingRecords.push(...records);
			this.pendingFlush ??= new Promise<{ offset: string }>((resolve, reject) => {
				this.resolvePending = resolve;
				this.rejectPending = reject;
			});
			if (this.pendingTimer === undefined) {
				if (Date.now() - this.lastFlushStartedAt >= CANONICAL_FLUSH_DELAY_MS) {
					queueMicrotask(() => {
						void this.flush().catch(() => {});
					});
				} else {
					this.pendingTimer = setTimeout(() => {
						void this.flush().catch(() => {});
					}, CANONICAL_FLUSH_DELAY_MS);
				}
			}
			return this.pendingFlush;
		} catch (error) {
			return Promise.reject(error);
		}
	}

	flush(): Promise<{ offset: string }> {
		try {
			this.assertActive();
			if (this.flushing) {
				if (this.pendingRecords.length === 0) return this.flushing;
				return this.flushing.then(() => this.flush());
			}
			if (this.pendingTimer) clearTimeout(this.pendingTimer);
			this.pendingTimer = undefined;
			if (this.pendingRecords.length === 0) {
				return Promise.resolve({ offset: this.offset });
			}
			this.lastFlushStartedAt = Date.now();
			const records = this.pendingRecords;
			const options = this.pendingOptions ?? {};
			const resolve = this.resolvePending;
			const reject = this.rejectPending;
			this.pendingRecords = [];
			this.pendingOptions = undefined;
			this.pendingFlush = undefined;
			this.resolvePending = undefined;
			this.rejectPending = undefined;
			const operation = this.appendBatch(records, options).then(
				(result) => {
					resolve?.(result);
					return result;
				},
				(error) => {
					reject?.(error);
					throw error;
				},
			);
			this.flushing = operation;
			void operation.then(
				() => {
					if (this.flushing === operation) this.flushing = undefined;
				},
				() => {},
			);
			return operation;
		} catch (error) {
			return Promise.reject(error);
		}
	}

	private appendBatch(
		records: readonly ConversationRecord[],
		options: ConversationAppendOptions,
	): Promise<{ offset: string }> {
		const operation = this.tail.then(async () => {
			this.assertActive();
			if (this.target.kind === 'harness')
				return this.appendThroughHarness(this.target, records, options);
			const claim = this.target.claim;
			const reduced = this.reducedState
				? reduceConversationRecords(
						this.reducedState,
						records,
						this.reducedState.recordsThroughOffset,
					)
				: undefined;
			const producerSequence = this.nextProducerSequence;
			const input = {
				path: this.path,
				producerId: claim.producerId,
				producerEpoch: claim.producerEpoch,
				incarnation: claim.incarnation,
				producerSequence,
				...(options.submission ? { submission: options.submission } : {}),
				records,
			};
			try {
				let result: { offset: string };
				try {
					result = await this.store.append(input);
				} catch (firstError) {
					try {
						result = await this.store.append(input);
					} catch {
						throw firstError;
					}
				}
				this.nextProducerSequence = producerSequence + 1;
				if (reduced) {
					reduced.recordsThroughOffset = result.offset;
					this.reducedState = reduced;
					this.foldHost.adoptState(reduced, claim.incarnation);
					// Durable fold checkpoint. The encode is synchronous (states
					// are never mutated after publication, so it reads a stable
					// snapshot); the store write floats off the append's critical
					// path — the batch is durable either way, and a lost
					// checkpoint just means the next cold load folds a longer
					// suffix.
					this.checkpointEvery(reduced, claim.incarnation);
				}
				return result;
			} catch (error) {
				throw this.fail(error);
			}
		});
		this.tail = operation.then(
			() => {},
			() => {},
		);
		return operation;
	}

	private async appendThroughHarness(
		target: Extract<WriterTarget, { kind: 'harness' }>,
		records: readonly ConversationRecord[],
		options: ConversationAppendOptions,
	) {
		const threaded = records.map((record) => ({
			...record,
			thread: record.conversationId,
		}));
		if (options.stage) {
			options.stage(threaded);
			return { offset: this.offset };
		}
		try {
			await target.append(threaded);
			const state = await this.foldHost.getStateAtHead();
			this.reducedState = state;
			this.checkpointEvery(state, target.incarnation);
			return { offset: state.recordsThroughOffset };
		} catch (error) {
			throw this.fail(error);
		}
	}

	/** Durable fold checkpoint every `FOLD_CHECKPOINT_INTERVAL` appends. A lost one only lengthens the next cold fold. */
	private checkpointEvery(state: ReducedInstanceState, incarnation: string) {
		this.batchesSinceFoldCheckpoint += 1;
		if (this.batchesSinceFoldCheckpoint < FOLD_CHECKPOINT_INTERVAL) return;
		this.batchesSinceFoldCheckpoint = 0;
		writeFoldCheckpoint(this.store, this.path, state, incarnation);
	}

	private assertActive(): void {
		if (this.lifecycle.status === 'failed') throw this.lifecycle.error;
	}

	private fail(error: unknown): unknown {
		if (this.lifecycle.status === 'failed') return this.lifecycle.error;
		this.lifecycle = { status: 'failed', error };
		this.foldHost.unpin();
		this.onFailed?.(this);
		if (this.pendingTimer) clearTimeout(this.pendingTimer);
		this.pendingTimer = undefined;
		this.pendingRecords = [];
		this.pendingOptions = undefined;
		const reject = this.rejectPending;
		this.pendingFlush = undefined;
		this.resolvePending = undefined;
		this.rejectPending = undefined;
		reject?.(error);
		return error;
	}

	async ensureChildConversation(input: {
		parent: ConversationRecordScope;
		child: Exclude<ConversationCreationInput, { kind: 'root' }>;
		ref: CanonicalChildSessionRef;
	}): Promise<{ offset: string }> {
		const state = await this.loadReducedState();
		const parent = state.conversations.get(input.parent.conversationId);
		if (
			!parent ||
			parent.harness !== input.parent.harness ||
			parent.session !== input.parent.session
		) {
			throw new Error('[flue] Canonical child parent is missing or conflicts with its scope.');
		}
		const existing = state.conversations.get(input.child.conversationId);
		const retained = parent.childConversations.get(input.child.conversationId);
		if (existing || retained) {
			if (
				!existing ||
				!retained ||
				existing.harness !== input.child.harness ||
				existing.session !== input.child.session ||
				existing.affinityKey !== input.child.affinityKey ||
				existing.parentConversationId !== input.parent.conversationId ||
				JSON.stringify(retained) !== JSON.stringify(input.ref)
			) {
				throw new Error('[flue] Canonical child conversation conflicts with retained topology.');
			}
			return { offset: state.recordsThroughOffset };
		}
		const timestamp = input.child.createdAt;
		return this.append([
			{
				v: 1,
				id: `record_conversation_created_${input.child.conversationId}`,
				type: 'conversation_created',
				conversationId: input.child.conversationId,
				harness: input.child.harness,
				session: input.child.session,
				timestamp,
				affinityKey: input.child.affinityKey,
				createdAt: input.child.createdAt,
				...(input.child.kind === 'task'
					? {
							kind: 'task' as const,
							parentConversationId: input.parent.conversationId,
							taskId: input.child.taskId,
							...(input.child.agent ? { agent: input.child.agent } : {}),
						}
					: {
							kind: 'action' as const,
							parentConversationId: input.parent.conversationId,
							actionInvocationId: input.child.actionInvocationId,
						}),
			},
			{
				v: 1,
				id: `record_child_retained_${input.parent.conversationId}_${input.child.conversationId}`,
				type: 'child_session_retained',
				conversationId: input.parent.conversationId,
				harness: input.parent.harness,
				session: input.parent.session,
				timestamp,
				child: input.ref,
			},
		]);
	}

	async ensureConversation(
		input: ConversationCreationInput & {
			timestamp?: string;
		},
	): Promise<{ offset: string }> {
		const state = await this.loadReducedState();
		const existing = state.conversations.get(input.conversationId);
		if (existing) {
			if (
				existing.harness !== input.harness ||
				existing.session !== input.session ||
				existing.affinityKey !== input.affinityKey ||
				existing.parentConversationId !== input.parentConversationId ||
				existing.taskId !== input.taskId ||
				existing.actionInvocationId !== input.actionInvocationId
			) {
				throw new Error(
					'[flue] Canonical conversation identity conflicts with the requested session.',
				);
			}
			return { offset: state.recordsThroughOffset };
		}
		const timestamp = input.timestamp ?? input.createdAt;
		return this.append([
			{
				...input,
				v: 1,
				id: `record_conversation_created_${input.conversationId}`,
				type: 'conversation_created',
				timestamp,
			},
		]);
	}
}

function sameAppendOptions(
	left: ConversationAppendOptions,
	right: ConversationAppendOptions,
): boolean {
	return (
		left.submission?.submissionId === right.submission?.submissionId &&
		left.submission?.attemptId === right.submission?.attemptId
	);
}
