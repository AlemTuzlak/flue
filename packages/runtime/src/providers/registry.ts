/**
 * The runtime's providers, by id. `app.ts`, the generated server entries, and
 * `flue run` register them, and `resolveModel` reads them for each model call.
 */
import { dynamicModel } from './catalog.ts';
import type { FlueModel, Provider } from './provider.ts';

const providers = new Map<string, Provider>();

/**
 * Register a provider, keyed by its `id`. A later call with the same `id`
 * replaces the earlier one, built-ins included.
 */
export function setProvider(provider: Provider) {
	providers.set(
		provider.id,
		provider.id === CLOUDFLARE_AI_GATEWAY_PROVIDER_ID
			? withAnthropicGatewayModelIds(provider)
			: provider,
	);
}

/** Whether a provider id is registered. */
export function hasProvider(providerId: string) {
	return providers.has(providerId);
}

/** The registered provider of an id. */
export function getProvider(providerId: string) {
	return providers.get(providerId);
}

/** Forget every provider. Test-only. */
export function resetProvidersForTests() {
	providers.clear();
}

/**
 * Register a built-in provider from its `@flue/runtime/providers/<id>` module
 * namespace. Generated entries call this, so the factory names stay out of
 * generated code. An id that is already registered is skipped, so `app.ts`
 * registrations win whatever the module order.
 */
export function registerBuiltinProviderModule(id: string, moduleNamespace: object) {
	if (hasProvider(id)) return;
	const factories = Object.entries(moduleNamespace).filter(
		(entry): entry is [string, () => unknown] =>
			entry[0].endsWith('Provider') && typeof entry[1] === 'function',
	);
	const factory = factories.length === 1 ? factories[0] : undefined;
	if (!factory) {
		const found = factories.map(([name]) => name).join(', ');
		throw new Error(
			`[flue] "@flue/runtime/providers/${id}" is not a single-provider module: ` +
				`expected exactly one exported \`*Provider\` factory, found ${factories.length === 0 ? 'none' : found}. ` +
				`Check the \`providers\` entry "${id}" in your flue() config.`,
		);
	}
	const provider = factory[1]();
	if (!isProvider(provider) || provider.id !== id) {
		const registered = isProvider(provider) ? provider.id : undefined;
		throw new Error(
			`[flue] "@flue/runtime/providers/${id}" registered provider ID ` +
				`"${registered}" instead of "${id}". Check the \`providers\` entry in your flue() config.`,
		);
	}
	setProvider(provider);
}

function isProvider(value: unknown): value is Provider {
	return (
		typeof value === 'object' &&
		value !== null &&
		'id' in value &&
		typeof value.id === 'string' &&
		'getModels' in value &&
		typeof value.getModels === 'function'
	);
}

/** A provider's models. A `getModels()` that throws reads as no models. */
function modelsOf(provider: Provider) {
	try {
		return provider.getModels();
	} catch {
		return [];
	}
}

// ─── Model resolution ───────────────────────────────────────────────────────

/**
 * Resolve a `provider-id/model-id` specifier against the registered
 * providers: a listed model, then a version-separator alias, then a dynamic
 * model when the provider has a template.
 */
// The return type is written out: declaration emit cannot name the dynamic-model marker.
export function resolveModel(modelSpecifier: string): FlueModel {
	const slash = modelSpecifier.indexOf('/');
	if (slash === -1) {
		throw new Error(
			`[flue] Invalid model specifier "${modelSpecifier}". ` +
				`Use the "provider-id/model-id" format (e.g. "anthropic/claude-haiku-4-5").`,
		);
	}
	const providerId = modelSpecifier.slice(0, slash);
	const modelId = modelSpecifier.slice(slash + 1);

	const provider = providers.get(providerId);
	if (!provider) {
		const registered = [...providers.keys()].sort();
		throw new Error(
			`[flue] Unknown provider "${providerId}" in model specifier "${modelSpecifier}". ` +
				(registered.length > 0
					? `Registered providers: ${registered.join(', ')}. `
					: 'No providers are registered. ') +
				`Include built-in providers via the \`providers\` option of the flue() Vite plugin, ` +
				`or register one with setProvider() in app.ts.`,
		);
	}
	if (modelId === '') {
		throw new Error(
			`[flue] Invalid model specifier "${modelSpecifier}". ` +
				`Provider "${providerId}" is registered, but no model ID was given. ` +
				`Use "${providerId}/<model-id>".`,
		);
	}

	const models = modelsOf(provider);
	const listed = models.find((model) => model.id === modelId);
	if (listed) return listed;

	const aliased = resolveVersionSeparatorAlias(providerId, modelId, models);
	if (aliased) return aliased;

	if (provider.dynamicModels) return dynamicModel(providerId, modelId, provider.dynamicModels);

	throw new Error(
		`[flue] Unknown model ID "${modelId}" for provider "${providerId}". ` +
			`Declared model IDs: ${listModelIds(models)}.`,
	);
}

function listModelIds(models: readonly FlueModel[]) {
	if (models.length === 0) return '(none)';
	const shown = models
		.slice(0, 8)
		.map((model) => model.id)
		.join(', ');
	return models.length > 8 ? `${shown}, … (${models.length} total)` : shown;
}

// ─── Version-separator aliases ──────────────────────────────────────────────

/**
 * A model id with `.` for every version separator between two digits, so
 * `claude-sonnet-4-6` and `claude-sonnet-4.6` compare equal.
 */
function versionSeparatorKey(modelId: string) {
	return modelId.replace(/(?<=\d)[-.](?=\d)/g, '.');
}

/** One warning per requested specifier: the alias was used. */
const warnedVersionSeparatorAliases = new Set<string>();

/** Reset the version-separator alias warnings. Test-only. */
export function resetVersionSeparatorAliasWarnForTests() {
	warnedVersionSeparatorAliases.clear();
}

/**
 * The one listed model whose id differs from `modelId` only in its version
 * separators (`-` or `.` between digits). Catalogs rename ids between these
 * forms, and a renamed id must not break every specifier that names the old
 * form. Two or more matches resolve to nothing, so the usual error reports
 * them.
 */
export function resolveVersionSeparatorAlias<TModel extends { id: string }>(
	providerId: string,
	modelId: string,
	models: readonly TModel[],
) {
	if (!/\d[-.]\d/.test(modelId)) return undefined;
	const key = versionSeparatorKey(modelId);
	const candidates = models.filter(
		(model) => model.id !== modelId && versionSeparatorKey(model.id) === key,
	);
	const [model] = candidates;
	if (candidates.length !== 1 || !model) return undefined;

	const requested = `${providerId}/${modelId}`;
	if (!warnedVersionSeparatorAliases.has(requested)) {
		warnedVersionSeparatorAliases.add(requested);
		console.warn(
			`[flue] Model "${requested}" is not in the provider's catalog; using ` +
				`"${providerId}/${model.id}", which differs only in version separators. ` +
				`Update the specifier to "${providerId}/${model.id}".`,
		);
	}
	return model;
}

// ─── Cloudflare AI Gateway Anthropic ids ────────────────────────────────────

const CLOUDFLARE_AI_GATEWAY_PROVIDER_ID = 'cloudflare-ai-gateway';

/**
 * Whether a model is served by AI Gateway's native Anthropic endpoint
 * (`…/{CLOUDFLARE_GATEWAY_ID}/anthropic`), which sends the model id to
 * Anthropic's Messages API unchanged.
 */
export function isAnthropicGatewayModel(model: { api: string; baseUrl: string }) {
	return model.api === 'anthropic-messages' && /\/anthropic\/?$/.test(model.baseUrl);
}

/**
 * The Anthropic API id for a gateway catalog id: dotted versions become
 * dashed (`claude-sonnet-4.6` → `claude-sonnet-4-6`). Anthropic accepts only
 * dashed ids and answers dotted ones with a 404.
 */
export function anthropicGatewayModelId(modelId: string) {
	return modelId.replace(/(?<=\d)\.(?=\d)/g, '-');
}

/**
 * The gateway's models with Anthropic ids on the native Anthropic endpoint,
 * keeping the first model of each id. Dotted specifiers still resolve
 * through the version-separator alias.
 */
export function withAnthropicGatewayIds<
	TModel extends { id: string; api: string; baseUrl: string },
>(models: readonly TModel[]) {
	const seen = new Set<string>();
	return models.flatMap((model) => {
		const id = isAnthropicGatewayModel(model) ? anthropicGatewayModelId(model.id) : model.id;
		if (seen.has(id)) return [];
		seen.add(id);
		return [id === model.id ? model : { ...model, id }];
	});
}

function withAnthropicGatewayModelIds(provider: Provider) {
	return {
		...provider,
		getModels: () => withAnthropicGatewayIds(provider.getModels()),
	};
}
