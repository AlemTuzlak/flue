/**
 * The durable TanStack harness host of one agent instance, on Flue's stores.
 */
import type { ModelMessage } from '@tanstack/ai';
import { createHarnessHost, defineHarness, type HarnessSession } from '@tanstack/ai-harness';
import { defineAIPersistence, type LeaseStore, type LogRecord } from '@tanstack/ai-persistence';
import { type AgentSubmissionStore, LEASE_DURATION_MS } from '../agent-execution-store.ts';
import type { HarnessLogRecord } from '../conversation-records.ts';
import { renderSignalMessage } from '../message-rendering.ts';
import type {
	ConversationStreamIdentity,
	ConversationStreamStore,
} from './conversation-stream-store.ts';
import { createFlueLeaseStore } from './harness-lease-store.ts';
import { createFlueLogStore } from './harness-log-store.ts';

/**
 * The durable harness host of one agent instance. Its log is the instance
 * stream at `path`, and its leases are the submission leases.
 *
 * Open each conversation of the instance as a thread with `logId: path`. An
 * append to another log fails, so no other stream gets the instance identity.
 * `ownerId` is the producer of the stream and the owner of the leases.
 * `isLive` goes to `createFlueLeaseStore`.
 *
 * Without `submissions`, the leases live in this process only: a lease is
 * alive from `acquire` to `release`. Use it where no other process can run
 * the instance (the local runtime).
 *
 * The host folds Flue's `signal` records into the model context of their
 * thread (see {@link projectFlueRecord}).
 *
 * @example
 * ```ts
 * const host = createInstanceHarnessHost({ streams, submissions, path, identity, ownerId });
 * const session = await host.open(harness, { threadId: conversationId, logId: path });
 * ```
 */
export function createInstanceHarnessHost(options: {
	streams: ConversationStreamStore;
	submissions?: AgentSubmissionStore;
	/** The agent stream path (see `agentStreamPath`). It is the log id. */
	path: string;
	identity: ConversationStreamIdentity;
	ownerId: string;
	isLive?: (inputId: string) => boolean;
}) {
	const { streams, submissions, path, identity, ownerId, isLive } = options;
	const log = createFlueLogStore(streams, {
		producerId: ownerId,
		identityFor: (logId) => {
			if (logId !== path) {
				throw new Error(
					`[flue] The harness host of "${path}" cannot write the log "${logId}". Open the thread with logId "${path}".`,
				);
			}
			return identity;
		},
	});
	const leases = submissions
		? createFlueLeaseStore(submissions, ownerId, { isLive })
		: createProcessLeaseStore();
	return createHarnessHost({
		persistence: defineAIPersistence({ stores: { log, leases } }),
		project: { record: projectFlueRecord, version: 'flue-signals-1' },
		coalesceMs: 1000,
		// The harness renews the lease three times in each lease period.
		lease: { ttlMs: LEASE_DURATION_MS, renewMs: LEASE_DURATION_MS / 3 },
	});
}

/** The host that {@link createInstanceHarnessHost} returns. */
export type InstanceHarnessHost = ReturnType<typeof createInstanceHarnessHost>;

/** An instance host, and the instance stream path that is its log. */
export interface InstanceHarnessBinding {
	host: InstanceHarnessHost;
	/** The instance stream path. Every thread of the host opens with it as `logId`. */
	logId: string;
}

/** Leases of this process: alive from `acquire` to `release`. */
function createProcessLeaseStore() {
	const held = new Set<string>();
	return {
		acquire: async (lease) => {
			held.add(lease.inputId);
		},
		renew: async (lease) => {
			held.add(lease.inputId);
		},
		release: async (lease) => {
			held.delete(lease.inputId);
		},
		isAlive: async (key) => held.has(key.inputId),
	} satisfies LeaseStore;
}

/** The recovery pair. A later step gives its text to the harness as `continueCutOff.note`. */
const UNPROJECTED_SIGNAL_TYPES: ReadonlySet<string> = new Set([
	'stream_interrupted',
	'stream_continued',
]);

/**
 * The host `project` fold of Flue's records: the records that change the
 * model context without a harness input. That is a `signal` record
 * (narration, advisories, `ctx.append`, a delivered signal), which adds one
 * user message with the rendered signal, as `buildConversationContext`
 * renders it. The recovery pair (`stream_interrupted`, `stream_continued`)
 * is left out. User inputs and assistant and tool messages are left out too:
 * the harness transcript has them from the prompt and the model calls.
 *
 * The fold is pure: the same record always adds the same message.
 */
export function projectFlueRecord(args: {
	messages: ReadonlyArray<ModelMessage>;
	record: LogRecord;
}) {
	const { messages, record } = args;
	if (record.type !== 'signal' || record.v !== 1) return undefined;
	const { signalType, content, tagName, attributes, messageId } = record;
	if (typeof signalType !== 'string' || typeof content !== 'string') return undefined;
	if (UNPROJECTED_SIGNAL_TYPES.has(signalType)) return undefined;
	const text = renderSignalMessage({
		role: 'signal',
		type: signalType,
		content,
		...(typeof tagName === 'string' ? { tagName } : {}),
		...(isStringRecord(attributes) ? { attributes } : {}),
		timestamp: 0,
	});
	const message: ModelMessage = {
		role: 'user',
		content: text,
		...(typeof messageId === 'string' ? { id: messageId } : {}),
	};
	return [...messages, message];
}

function isStringRecord(value: unknown): value is Record<string, string> {
	return (
		typeof value === 'object' &&
		value !== null &&
		Object.values(value).every((entry) => typeof entry === 'string')
	);
}

/** The thread of the host records that no conversation's own turn appends. */
const RECORDS_THREAD = 'flue:records';

/**
 * Append host records to the log of `binding`, through one thread that only
 * appends. Each record names its own thread with `thread`. Pass the result
 * as the `append` of `ConversationRecordWriter.overHarness`.
 *
 * The appends share the log's one writer with every turn of the host, so a
 * record lands in the next log batch, after the events that wait.
 */
export function createHostRecordAppend(binding: InstanceHarnessBinding) {
	let session: Promise<HarnessSession> | undefined;
	return async (records: readonly HarnessLogRecord[]) => {
		session ??= binding.host.open(defineHarness({ name: 'flue/records' }), {
			threadId: RECORDS_THREAD,
			logId: binding.logId,
		});
		await (await session).append(records);
	};
}
