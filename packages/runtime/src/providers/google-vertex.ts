import * as catalog from '@tanstack/ai-models/google-vertex';
import { builtinProvider } from './builtins.ts';

/** The Google Vertex AI provider, with the `@tanstack/ai-models` catalog. */
export function googleVertexProvider() {
	return builtinProvider(catalog);
}
