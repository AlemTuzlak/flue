import * as catalog from '@tanstack/ai-models/ant-ling';
import { builtinProvider } from './builtins.ts';

/** The Ant Ling provider, with the `@tanstack/ai-models` catalog. */
export function antLingProvider() {
	return builtinProvider(catalog);
}
