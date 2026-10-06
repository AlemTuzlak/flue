import * as catalog from '@tanstack/ai-models/minimax-cn';
import { builtinProvider } from './builtins.ts';

/** The MiniMax (China) provider, with the `@tanstack/ai-models` catalog. */
export function minimaxCnProvider() {
	return builtinProvider(catalog);
}
