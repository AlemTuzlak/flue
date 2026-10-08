import * as catalog from '@tanstack/ai-models/xai';
import { builtinProvider } from './builtins.ts';

/** The xAI provider, with the `@tanstack/ai-models` catalog. */
export function xaiProvider() {
	return builtinProvider(catalog);
}
