import * as catalog from '@tanstack/ai-models/baseten';
import { builtinProvider } from './builtins.ts';

/** The Baseten provider, with the `@tanstack/ai-models` catalog. */
export function basetenProvider() {
	return builtinProvider(catalog);
}
