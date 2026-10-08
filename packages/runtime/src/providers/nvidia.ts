import * as catalog from '@tanstack/ai-models/nvidia';
import { builtinProvider } from './builtins.ts';

/** The NVIDIA provider, with the `@tanstack/ai-models` catalog. */
export function nvidiaProvider() {
	return builtinProvider(catalog);
}
