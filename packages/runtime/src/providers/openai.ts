import * as catalog from '@tanstack/ai-models/openai';
import { builtinProvider } from './builtins.ts';

/** The OpenAI provider, with the `@tanstack/ai-models` catalog. */
export function openaiProvider() {
	return builtinProvider(catalog);
}
