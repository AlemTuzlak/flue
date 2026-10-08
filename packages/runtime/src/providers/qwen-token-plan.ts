import * as catalog from '@tanstack/ai-models/qwen-token-plan';
import { builtinProvider } from './builtins.ts';

/** The Qwen Token Plan provider, with the `@tanstack/ai-models` catalog. */
export function qwenTokenPlanProvider() {
	return builtinProvider(catalog);
}
