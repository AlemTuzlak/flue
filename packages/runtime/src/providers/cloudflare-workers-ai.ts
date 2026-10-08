import * as catalog from '@tanstack/ai-models/cloudflare-workers-ai';
import { builtinProvider } from './builtins.ts';

/** The Cloudflare Workers AI provider, with the `@tanstack/ai-models` catalog. */
export function cloudflareWorkersAIProvider() {
	return builtinProvider(catalog);
}
