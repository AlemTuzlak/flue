import * as catalog from '@tanstack/ai-models/huggingface';
import { builtinProvider } from './builtins.ts';

/** The Hugging Face provider, with the `@tanstack/ai-models` catalog. */
export function huggingfaceProvider() {
	return builtinProvider(catalog);
}
