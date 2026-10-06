import * as catalog from '@tanstack/ai-models/deepseek';
import { builtinProvider } from './builtins.ts';

/** The DeepSeek provider, with the `@tanstack/ai-models` catalog. */
export function deepseekProvider() {
	return builtinProvider(catalog);
}
