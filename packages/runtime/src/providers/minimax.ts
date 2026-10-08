import * as catalog from '@tanstack/ai-models/minimax';
import { builtinProvider } from './builtins.ts';

/** The MiniMax provider, with the `@tanstack/ai-models` catalog. */
export function minimaxProvider() {
	return builtinProvider(catalog);
}
