import * as catalog from '@tanstack/ai-models/cloudflare-ai-gateway';
import { builtinProvider } from './builtins.ts';

/** The Cloudflare AI Gateway provider, with the `@tanstack/ai-models` catalog. */
export function cloudflareAIGatewayProvider() {
	return builtinProvider(catalog);
}
