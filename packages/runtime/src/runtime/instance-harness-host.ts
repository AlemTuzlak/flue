/**
 * The durable TanStack harness host of one agent instance, on Flue's stores.
 */
import { createHarnessHost } from '@tanstack/ai-harness';
import { defineAIPersistence } from '@tanstack/ai-persistence';
import { type AgentSubmissionStore, LEASE_DURATION_MS } from '../agent-execution-store.ts';
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
 * @example
 * ```ts
 * const host = createInstanceHarnessHost({ streams, submissions, path, identity, ownerId });
 * const session = await host.open(harness, { threadId: conversationId, logId: path });
 * ```
 */
export function createInstanceHarnessHost(options: {
	streams: ConversationStreamStore;
	submissions: AgentSubmissionStore;
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
	return createHarnessHost({
		persistence: defineAIPersistence({
			stores: { log, leases: createFlueLeaseStore(submissions, ownerId, { isLive }) },
		}),
		coalesceMs: 1000,
		// The harness renews the lease three times in each lease period.
		lease: { ttlMs: LEASE_DURATION_MS, renewMs: LEASE_DURATION_MS / 3 },
	});
}
