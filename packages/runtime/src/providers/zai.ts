import * as catalog from '@tanstack/ai-models/zai';
import { builtinProvider } from './builtins.ts';

/** The Z.AI provider, with the `@tanstack/ai-models` catalog. */
export function zaiProvider() {
	return builtinProvider(catalog);
}
