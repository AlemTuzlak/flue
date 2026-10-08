import * as catalog from '@tanstack/ai-models/kimi-coding';
import { builtinProvider } from './builtins.ts';

/** The Kimi For Coding provider, with the `@tanstack/ai-models` catalog. */
export function kimiCodingProvider() {
	return builtinProvider(catalog);
}
