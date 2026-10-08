import * as catalog from '@tanstack/ai-models/meta';
import { builtinProvider } from './builtins.ts';

/** The Meta provider, with the `@tanstack/ai-models` catalog. */
export function metaProvider() {
	return builtinProvider(catalog);
}
