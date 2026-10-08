import * as catalog from '@tanstack/ai-models/xiaomi-token-plan-sgp';
import { builtinProvider } from './builtins.ts';

/** The Xiaomi MiMo Token Plan (Singapore) provider, with the `@tanstack/ai-models` catalog. */
export function xiaomiTokenPlanSgpProvider() {
	return builtinProvider(catalog);
}
