import * as catalog from '@tanstack/ai-models/fireworks';
import { builtinProvider } from './builtins.ts';

/** The Fireworks AI provider, with the `@tanstack/ai-models` catalog. */
export function fireworksProvider() {
	return builtinProvider(catalog);
}
