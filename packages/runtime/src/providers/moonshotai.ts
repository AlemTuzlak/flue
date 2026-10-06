import * as catalog from '@tanstack/ai-models/moonshotai';
import { builtinProvider } from './builtins.ts';

/** The Moonshot AI provider, with the `@tanstack/ai-models` catalog. */
export function moonshotaiProvider() {
	return builtinProvider(catalog);
}
