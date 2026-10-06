import * as catalog from '@tanstack/ai-models/anthropic';
import { builtinProvider } from './builtins.ts';

/** The Anthropic provider, with the `@tanstack/ai-models` catalog. */
export function anthropicProvider() {
	return builtinProvider(catalog);
}
