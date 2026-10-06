import * as catalog from '@tanstack/ai-models/moonshotai-cn';
import { builtinProvider } from './builtins.ts';

/** The Moonshot AI (China) provider, with the `@tanstack/ai-models` catalog. */
export function moonshotaiCnProvider() {
	return builtinProvider(catalog);
}
