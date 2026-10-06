import * as catalog from '@tanstack/ai-models/together';
import { builtinProvider } from './builtins.ts';

/** The Together AI provider, with the `@tanstack/ai-models` catalog. */
export function togetherProvider() {
	return builtinProvider(catalog);
}
