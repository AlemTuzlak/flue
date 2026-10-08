import * as catalog from '@tanstack/ai-models/xiaomi';
import { builtinProvider } from './builtins.ts';

/** The Xiaomi MiMo provider, with the `@tanstack/ai-models` catalog. */
export function xiaomiProvider() {
	return builtinProvider(catalog);
}
