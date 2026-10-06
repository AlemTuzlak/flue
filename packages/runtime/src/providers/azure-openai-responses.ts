import * as catalog from '@tanstack/ai-models/azure-openai-responses';
import { builtinProvider } from './builtins.ts';

/** The Azure OpenAI provider, with the `@tanstack/ai-models` catalog. */
export function azureOpenAIResponsesProvider() {
	return builtinProvider(catalog);
}
