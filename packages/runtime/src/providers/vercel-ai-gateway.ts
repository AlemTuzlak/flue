import * as catalog from '@tanstack/ai-models/vercel-ai-gateway';
import { builtinProvider } from './builtins.ts';

/** The Vercel AI Gateway provider, with the `@tanstack/ai-models` catalog. */
export function vercelAIGatewayProvider() {
	return builtinProvider(catalog);
}
