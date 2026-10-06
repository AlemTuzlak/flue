import * as catalog from '@tanstack/ai-models/opencode';
import { builtinProvider } from './builtins.ts';

/** The OpenCode Zen provider, with the `@tanstack/ai-models` catalog. */
export function opencodeProvider() {
	return builtinProvider(catalog);
}
