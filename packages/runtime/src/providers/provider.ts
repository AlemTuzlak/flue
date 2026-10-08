/**
 * Flue's provider: a set of `@tanstack/ai-models` records plus the auth that
 * a request needs. The model's `api` picks the TanStack adapter, unless the
 * provider builds its own with `createAdapter`.
 */
import type { AnyTextAdapter } from '@tanstack/ai';
import type { ModelRecord, WireApi } from '@tanstack/ai-models';

/** One model of a provider. `api` is a TanStack wire API, or a name that the provider's `createAdapter` reads. */
export type FlueModel = Omit<ModelRecord, 'api'> & {
	api: WireApi | (string & {});
};

/** Request headers. `null` removes a header that the adapter would send. */
export type ProviderHeaders = Readonly<Record<string, string | null>>;

/** Auth for one request. */
export interface ModelAuth {
	apiKey?: string;
	headers?: ProviderHeaders;
	baseUrl?: string;
}

/** What a provider's auth resolved to. `env` holds provider config, such as a Cloudflare account id. */
export interface AuthResult {
	auth: ModelAuth;
	env?: Readonly<Record<string, string>>;
	/** Where the credential came from, for example `ANTHROPIC_API_KEY`. */
	source?: string;
}

/** What an auth resolver can read. */
export interface AuthContext {
	/** An environment variable. A blank value reads as `undefined`. */
	env(name: string): Promise<string | undefined>;
	/** Whether a file exists. A leading `~` is the home directory. Always `false` without a file system. */
	fileExists(path: string): Promise<boolean>;
}

/** API key auth. `resolve` runs for each request and returns `undefined` when the provider is not configured. */
export interface ApiKeyAuth {
	/** Display name, for example `Anthropic API key`. */
	name: string;
	resolve(input: { ctx: AuthContext; signal: AbortSignal }): Promise<AuthResult | undefined>;
}

export interface ProviderAuth {
	apiKey: ApiKeyAuth;
}

/** The endpoint for model ids that a provider serves but does not list. */
export interface DynamicModelTemplate {
	api: FlueModel['api'];
	baseUrl: string;
}

/** What a provider gets to build the adapter for one model call. */
export interface ModelAdapterOptions {
	/** `undefined` when the provider's auth found no credential. */
	auth: AuthResult | undefined;
	/** The conversation's prompt cache key. */
	promptCacheKey: string;
	promptCache?: 'none' | 'short' | 'long';
	fetch?: typeof fetch;
}

export interface Provider {
	readonly id: string;
	readonly name: string;
	readonly baseUrl?: string;
	readonly headers?: ProviderHeaders;
	readonly auth: ProviderAuth;
	/** The models the provider lists. */
	getModels(): readonly FlueModel[];
	/** Serve model ids that `getModels()` does not list, with zero metadata. */
	readonly dynamicModels?: DynamicModelTemplate;
	/** Build the adapter yourself, for a wire protocol that Flue does not map. */
	createAdapter?(model: FlueModel, options: ModelAdapterOptions): AnyTextAdapter;
}

export interface CreateProviderConfig {
	id: string;
	/** Display name. The default is `id`. */
	name?: string;
	baseUrl?: string;
	headers?: ProviderHeaders;
	auth: ProviderAuth;
	models: readonly FlueModel[];
	dynamicModels?: DynamicModelTemplate;
	createAdapter?: Provider['createAdapter'];
}

/**
 * Build a provider for `setProvider()`.
 *
 * ```ts
 * setProvider(
 *   createProvider({
 *     id: 'ollama',
 *     auth: { apiKey: { name: 'Ollama (keyless)', resolve: async () => ({ auth: {} }) } },
 *     models: [{ id: 'llama3.1:8b', api: 'openai-completions', baseUrl: 'http://localhost:11434/v1', ... }],
 *   }),
 * );
 * ```
 */
export function createProvider(config: CreateProviderConfig): Provider {
	const { models, name, ...rest } = config;
	return { ...rest, name: name ?? config.id, getModels: () => models };
}

/** API key auth from the first of `envVars` that is set. */
export function envApiKeyAuth(name: string, envVars: readonly string[]): ApiKeyAuth {
	return {
		name,
		resolve: async ({ ctx, signal }) => {
			for (const envVar of envVars) {
				signal.throwIfAborted();
				const apiKey = await ctx.env(envVar);
				if (apiKey) return { auth: { apiKey }, source: envVar };
			}
			return undefined;
		},
	};
}

// A variable specifier, so bundlers for Workers do not try to resolve Node builtins.
const importNodeModule = (specifier: string) => import(/* @vite-ignore */ specifier);

/** Reads `process.env` and the file system, when there are any. */
export const defaultAuthContext: AuthContext = {
	async env(name) {
		const value = globalThis.process?.env?.[name];
		return typeof value === 'string' && value.trim().length > 0 ? value : undefined;
	},
	async fileExists(path) {
		try {
			const fs: typeof import('node:fs/promises') = await importNodeModule('node:fs/promises');
			const os: typeof import('node:os') = await importNodeModule('node:os');
			await fs.access(path.startsWith('~') ? os.homedir() + path.slice(1) : path);
			return true;
		} catch {
			return false;
		}
	},
};

/** Resolve a provider's auth for one request. */
export function resolveProviderAuth(provider: Provider, signal: AbortSignal) {
	return provider.auth.apiKey.resolve({ ctx: defaultAuthContext, signal });
}
