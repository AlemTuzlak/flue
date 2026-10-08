import * as catalog from '@tanstack/ai-models/groq';
import { builtinProvider } from './builtins.ts';

/** The Groq provider, with the `@tanstack/ai-models` catalog. */
export function groqProvider() {
	return builtinProvider(catalog);
}
