import * as catalog from '@tanstack/ai-models/xiaomi-token-plan-cn';
import { builtinProvider } from './builtins.ts';

/** The Xiaomi MiMo Token Plan (China) provider, with the `@tanstack/ai-models` catalog. */
export function xiaomiTokenPlanCnProvider() {
	return builtinProvider(catalog);
}
