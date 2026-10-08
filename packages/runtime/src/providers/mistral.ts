import * as catalog from '@tanstack/ai-models/mistral';
import { builtinProvider } from './builtins.ts';

/** The Mistral provider, with the `@tanstack/ai-models` catalog. */
export function mistralProvider() {
	return builtinProvider(catalog);
}
