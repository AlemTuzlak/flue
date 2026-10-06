import * as catalog from '@tanstack/ai-models/amazon-bedrock';
import { builtinProvider } from './builtins.ts';

/** The Amazon Bedrock provider, with the `@tanstack/ai-models` catalog. */
export function amazonBedrockProvider() {
	return builtinProvider(catalog);
}
