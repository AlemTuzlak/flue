import * as catalog from '@tanstack/ai-models/opencode-go';
import { builtinProvider } from './builtins.ts';

/** The OpenCode Go provider, with the `@tanstack/ai-models` catalog. */
export function opencodeGoProvider() {
	return builtinProvider(catalog);
}
