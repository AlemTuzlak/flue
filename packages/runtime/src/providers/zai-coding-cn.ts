import * as catalog from '@tanstack/ai-models/zai-coding-cn';
import { builtinProvider } from './builtins.ts';

/** The Zhipu AI Coding Plan provider, with the `@tanstack/ai-models` catalog. */
export function zaiCodingCnProvider() {
	return builtinProvider(catalog);
}
