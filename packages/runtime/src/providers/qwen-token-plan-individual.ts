import * as catalog from '@tanstack/ai-models/qwen-token-plan-individual';
import { builtinProvider } from './builtins.ts';

/** The Qwen Token Plan (Individual) provider, with the `@tanstack/ai-models` catalog. */
export function qwenTokenPlanIndividualProvider() {
	return builtinProvider(catalog);
}
