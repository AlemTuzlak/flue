/**
 * Internal helpers for aggregating `PromptUsage`. Not re-exported from the
 * public @flue/runtime entry — they're an implementation detail of how prompt(),
 * skill(), task() and compaction roll up token + cost figures.
 *
 * Kept in their own module to share between `session.ts` (per-call
 * aggregation across the active path) and `session.ts`'s compaction
 * persistence path (normalizing provider usage into our `PromptUsage`
 * before storing on a canonical compaction record).
 */
import type { TokenUsage } from '@tanstack/ai';
import type { ModelCostRates } from '@tanstack/ai-models';
import type { Usage } from './llm-types.ts';
import { toFlueUsage } from './model-messages.ts';
import type { PromptUsage } from './types.ts';

/** All-zero `PromptUsage`. Identity element for `addUsage`. */
export function emptyUsage(): PromptUsage {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

/**
 * Field-wise sum of two `PromptUsage` values, including the nested `cost`
 * sub-object. Returns a fresh object; neither argument is mutated.
 */
export function addUsage(a: PromptUsage, b: PromptUsage): PromptUsage {
	return {
		input: a.input + b.input,
		output: a.output + b.output,
		cacheRead: a.cacheRead + b.cacheRead,
		cacheWrite: a.cacheWrite + b.cacheWrite,
		totalTokens: a.totalTokens + b.totalTokens,
		cost: {
			input: a.cost.input + b.cost.input,
			output: a.cost.output + b.cost.output,
			cacheRead: a.cost.cacheRead + b.cost.cacheRead,
			cacheWrite: a.cost.cacheWrite + b.cost.cacheWrite,
			total: a.cost.total + b.cost.total,
		},
	};
}

/**
 * Convert a message's `Usage`, or TanStack's `TokenUsage` priced with `cost`,
 * into Flue's public `PromptUsage`. Going through this normalizer keeps the
 * public type apart from the message type. Returns `undefined` when the input
 * is `undefined`.
 */
export function fromProviderUsage(
	providerUsage: Usage | TokenUsage | undefined,
	cost?: ModelCostRates,
): PromptUsage | undefined {
	if (!providerUsage) return undefined;
	const usage = 'promptTokens' in providerUsage ? toFlueUsage(providerUsage, cost) : providerUsage;
	return {
		input: usage.input,
		output: usage.output,
		cacheRead: usage.cacheRead,
		cacheWrite: usage.cacheWrite,
		totalTokens: usage.totalTokens,
		cost: {
			input: usage.cost.input,
			output: usage.cost.output,
			cacheRead: usage.cost.cacheRead,
			cacheWrite: usage.cost.cacheWrite,
			total: usage.cost.total,
		},
	};
}
