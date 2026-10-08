import * as catalog from '@tanstack/ai-models/qwen-token-plan-cn';
import { builtinProvider } from './builtins.ts';

/** The Qwen Token Plan (China) provider, with the `@tanstack/ai-models` catalog. */
export function qwenTokenPlanCnProvider() {
	return builtinProvider(catalog);
}
