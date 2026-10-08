import * as catalog from '@tanstack/ai-models/openrouter';
import { builtinProvider } from './builtins.ts';

/** The OpenRouter provider, with the `@tanstack/ai-models` catalog. */
export function openrouterProvider() {
	return builtinProvider(catalog);
}
