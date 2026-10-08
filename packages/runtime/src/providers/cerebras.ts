import * as catalog from '@tanstack/ai-models/cerebras';
import { builtinProvider } from './builtins.ts';

/** The Cerebras provider, with the `@tanstack/ai-models` catalog. */
export function cerebrasProvider() {
	return builtinProvider(catalog);
}
