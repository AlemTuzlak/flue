/**
 * The built-in providers: one `@tanstack/ai-models` catalog entry each, with
 * pi's auth rules. Most read one API key variable. The providers below also
 * need config, which their auth returns in `env` for the adapter.
 */
import type { ModelRecord, ProviderRecord } from '@tanstack/ai-models';
import { type ApiKeyAuth, type AuthContext, createProvider, envApiKeyAuth } from './provider.ts';

/** A provider's catalog module, such as `@tanstack/ai-models/anthropic`. */
export interface ProviderCatalog {
	provider: ProviderRecord;
	models: readonly ModelRecord[];
}

/** The provider of a catalog module, with its built-in auth. */
export function builtinProvider(catalog: ProviderCatalog) {
	const { provider, models } = catalog;
	return createProvider({
		id: provider.id,
		name: provider.name,
		...(provider.baseUrl ? { baseUrl: provider.baseUrl } : {}),
		auth: { apiKey: builtinAuth(provider) },
		models,
	});
}

function builtinAuth(provider: ProviderRecord) {
	switch (provider.id) {
		case 'anthropic':
			return anthropicAuth;
		case 'google-vertex':
			return vertexAuth;
		case 'amazon-bedrock':
			return bedrockAuth;
		case 'azure-openai-responses':
			return withConfigEnv(
				envApiKeyAuth('Azure OpenAI API key', ['AZURE_OPENAI_API_KEY']),
				AZURE_CONFIG_ENV,
			);
		case 'cloudflare-ai-gateway':
			return cloudflareAuth('ai-gateway');
		case 'cloudflare-workers-ai':
			return cloudflareAuth('workers-ai');
		default:
			return envApiKeyAuth(`${provider.name} API key`, provider.env.flat());
	}
}

/** Read `name`, and stop when the request is aborted. */
async function readEnv(ctx: AuthContext, signal: AbortSignal, name: string) {
	signal.throwIfAborted();
	return ctx.env(name);
}

/** The variables of `names` that are set. */
async function readConfigEnv(ctx: AuthContext, signal: AbortSignal, names: readonly string[]) {
	const env: Record<string, string> = {};
	for (const name of names) {
		const value = await readEnv(ctx, signal, name);
		if (value) env[name] = value;
	}
	return env;
}

/** Add the set variables of `names` to what `auth` resolves. */
function withConfigEnv(auth: ApiKeyAuth, names: readonly string[]) {
	return {
		name: auth.name,
		resolve: async (input) => {
			const result = await auth.resolve(input);
			if (!result) return undefined;
			const env = await readConfigEnv(input.ctx, input.signal, names);
			return { ...result, env: { ...result.env, ...env } };
		},
	} satisfies ApiKeyAuth;
}

const AZURE_CONFIG_ENV = [
	'AZURE_OPENAI_BASE_URL',
	'AZURE_OPENAI_RESOURCE_NAME',
	'AZURE_OPENAI_API_VERSION',
	'AZURE_OPENAI_DEPLOYMENT_NAME_MAP',
];

/** `ANTHROPIC_AUTH_TOKEN` goes as a bearer header; an OAuth token or an API key goes as the key. */
const anthropicAuth: ApiKeyAuth = {
	name: 'Anthropic API key',
	resolve: async ({ ctx, signal }) => {
		const authToken = await readEnv(ctx, signal, 'ANTHROPIC_AUTH_TOKEN');
		if (authToken) {
			return {
				auth: { headers: { Authorization: `Bearer ${authToken}` } },
				source: 'ANTHROPIC_AUTH_TOKEN',
			};
		}
		for (const envVar of ['ANTHROPIC_OAUTH_TOKEN', 'ANTHROPIC_API_KEY']) {
			const apiKey = await readEnv(ctx, signal, envVar);
			if (apiKey) return { auth: { apiKey }, source: envVar };
		}
		return undefined;
	},
};

const VERTEX_ADC_PATH = '~/.config/gcloud/application_default_credentials.json';

/** An API key, or Application Default Credentials with a project and a location. */
const vertexAuth: ApiKeyAuth = {
	name: 'Google Cloud credentials',
	resolve: async ({ ctx, signal }) => {
		const apiKey = await readEnv(ctx, signal, 'GOOGLE_CLOUD_API_KEY');
		if (apiKey) return { auth: { apiKey }, source: 'GOOGLE_CLOUD_API_KEY' };
		const adcPath = await readEnv(ctx, signal, 'GOOGLE_APPLICATION_CREDENTIALS');
		const hasCredentials = await ctx.fileExists(adcPath ?? VERTEX_ADC_PATH);
		const project =
			(await readEnv(ctx, signal, 'GOOGLE_CLOUD_PROJECT')) ??
			(await readEnv(ctx, signal, 'GCLOUD_PROJECT'));
		const location = await readEnv(ctx, signal, 'GOOGLE_CLOUD_LOCATION');
		if (!hasCredentials || !project || !location) return undefined;
		return {
			auth: {},
			env: { GOOGLE_CLOUD_PROJECT: project, GOOGLE_CLOUD_LOCATION: location },
			source: 'gcloud application default credentials',
		};
	},
};

const BEDROCK_CONFIG_ENV = [
	'AWS_REGION',
	'AWS_DEFAULT_REGION',
	'AWS_PROFILE',
	'AWS_BEARER_TOKEN_BEDROCK',
	'AWS_BEDROCK_SKIP_AUTH',
	'AWS_BEDROCK_FORCE_HTTP1',
];

/** A bearer token, or any source of the AWS SDK's default credential chain. */
const bedrockAuth: ApiKeyAuth = {
	name: 'AWS credentials or bearer token',
	resolve: async ({ ctx, signal }) => {
		const source = await bedrockCredentialSource(ctx, signal);
		if (!source) return undefined;
		return {
			auth: {},
			env: await readConfigEnv(ctx, signal, BEDROCK_CONFIG_ENV),
			source,
		};
	},
};

async function bedrockCredentialSource(ctx: AuthContext, signal: AbortSignal) {
	if (await readEnv(ctx, signal, 'AWS_BEARER_TOKEN_BEDROCK')) return 'AWS_BEARER_TOKEN_BEDROCK';
	if (await readEnv(ctx, signal, 'AWS_PROFILE')) return 'AWS_PROFILE';
	const hasAccessKeys =
		(await readEnv(ctx, signal, 'AWS_ACCESS_KEY_ID')) &&
		(await readEnv(ctx, signal, 'AWS_SECRET_ACCESS_KEY'));
	if (hasAccessKeys) return 'AWS access keys';
	if (await readEnv(ctx, signal, 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI')) return 'ECS task role';
	if (await readEnv(ctx, signal, 'AWS_CONTAINER_CREDENTIALS_FULL_URI')) return 'ECS task role';
	if (await readEnv(ctx, signal, 'AWS_WEB_IDENTITY_TOKEN_FILE')) return 'web identity token';
	return undefined;
}

/**
 * `CLOUDFLARE_API_KEY` and `CLOUDFLARE_ACCOUNT_ID`, plus `CLOUDFLARE_GATEWAY_ID`
 * for AI Gateway. The gateway takes the key in `cf-aig-authorization` and no
 * provider key.
 */
function cloudflareAuth(kind: 'ai-gateway' | 'workers-ai') {
	return {
		name: 'Cloudflare API key',
		resolve: async ({ ctx, signal }) => {
			const apiKey = await readEnv(ctx, signal, 'CLOUDFLARE_API_KEY');
			const accountId = await readEnv(ctx, signal, 'CLOUDFLARE_ACCOUNT_ID');
			const gatewayId =
				kind === 'ai-gateway' ? await readEnv(ctx, signal, 'CLOUDFLARE_GATEWAY_ID') : undefined;
			if (!apiKey || !accountId || (kind === 'ai-gateway' && !gatewayId)) return undefined;
			const env = {
				CLOUDFLARE_ACCOUNT_ID: accountId,
				...(gatewayId ? { CLOUDFLARE_GATEWAY_ID: gatewayId } : {}),
			};
			if (kind === 'workers-ai') return { auth: { apiKey }, env, source: 'CLOUDFLARE_API_KEY' };
			return {
				auth: {
					headers: {
						'cf-aig-authorization': `Bearer ${apiKey}`,
						Authorization: null,
						'x-api-key': null,
					},
				},
				env,
				source: 'CLOUDFLARE_API_KEY',
			};
		},
	} satisfies ApiKeyAuth;
}
