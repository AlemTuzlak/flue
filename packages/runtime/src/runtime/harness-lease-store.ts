/**
 * The TanStack harness `LeaseStore` over Flue's submission leases. A harness
 * input id is the Flue submission id: the coordinator claims the submission
 * (owner and lease expiry), then prompts the harness with it.
 */
import type { LeaseStore } from '@tanstack/ai-persistence';
import type { AgentSubmissionStore } from '../agent-execution-store.ts';

/**
 * A `LeaseStore` whose leases are the submission leases of `ownerId`.
 * Taking and renewing a lease renew the submission's lease. The submission's
 * settlement ends it, so `release` does nothing.
 *
 * A lease is alive while the submission runs and its lease expiry is in the
 * future. With `isLive`, a lease is alive only while `isLive` returns true.
 */
export function createFlueLeaseStore(
	submissions: AgentSubmissionStore,
	ownerId: string,
	options: {
		/**
		 * True while this process runs the attempt of the submission. Use it
		 * where the submission rows have no lease expiry (Cloudflare). With
		 * it, the store does not read the lease expiry of the row.
		 */
		isLive?: (inputId: string) => boolean;
	} = {},
) {
	const { isLive } = options;
	return {
		acquire: (lease) => submissions.renewLeases(ownerId, [lease.inputId]),
		renew: (lease) => submissions.renewLeases(ownerId, [lease.inputId]),
		release: async () => {},
		isAlive: async (key) => {
			if (isLive) return isLive(key.inputId);
			const submission = await submissions.getSubmission(key.inputId);
			const isRunning = submission?.status === 'running' || submission?.status === 'terminalizing';
			return isRunning && submission.leaseExpiresAt > Date.now();
		},
	} satisfies LeaseStore;
}
