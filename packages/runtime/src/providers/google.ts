import * as catalog from '@tanstack/ai-models/google';
import { builtinProvider } from './builtins.ts';

/** The Google provider, with the `@tanstack/ai-models` catalog. */
export function googleProvider() {
	return builtinProvider(catalog);
}
