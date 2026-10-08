/**
 * The default provider set: every built-in. A separate module, so the whole
 * catalog enters only a build that uses the default. Generated entries with a
 * `providers` list import the listed `@flue/runtime/providers/<id>` modules.
 */
import { getModels, getProviders } from '@tanstack/ai-models';
import { builtinProvider } from './builtins.ts';
import { hasProvider, setProvider } from './registry.ts';

/** Every built-in provider. */
export function builtinProviders() {
	return getProviders().map((provider) =>
		builtinProvider({ provider, models: getModels(provider.id) }),
	);
}

/** Register every built-in provider whose id is not registered yet, so `setProvider()` calls win. */
export function registerDefaultProviders() {
	for (const provider of builtinProviders()) {
		if (!hasProvider(provider.id)) setProvider(provider);
	}
}
