import * as catalog from '@tanstack/ai-models/xiaomi-token-plan-ams';
import { builtinProvider } from './builtins.ts';

/** The Xiaomi MiMo Token Plan (Amsterdam) provider, with the `@tanstack/ai-models` catalog. */
export function xiaomiTokenPlanAmsProvider() {
	return builtinProvider(catalog);
}
