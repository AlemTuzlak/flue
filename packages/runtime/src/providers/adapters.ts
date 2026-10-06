/**
 * The TanStack adapter for one Flue model call. The model's `api` picks the
 * adapter. The record and the provider's auth give it the base URL, the
 * headers, the key, and the reasoning data. The request rules that pi applied
 * and the adapters do not are here too: the session headers, the output token
 * cap, and `store: false`.
 */
import type { AnyTextAdapter, TextOptions } from '@tanstack/ai';
import { anthropicText } from '@tanstack/ai-anthropic';
import { createBedrockConverse } from '@tanstack/ai-bedrock';
import { createGeminiChat } from '@tanstack/ai-gemini';
import { createMistralText } from '@tanstack/ai-mistral';
import { modelReasoning, type WireApi } from '@tanstack/ai-models';
import { azureOpenaiText, createOpenaiChat } from '@tanstack/ai-openai';
import { openaiCompatibleText } from '@tanstack/ai-openai/compatible';
import type {
	AuthResult,
	FlueModel,
	ModelAdapterOptions,
	ModelAuth,
	Provider,
} from './provider.ts';

/**
 * The adapter for one call of `model`. A provider with `createAdapter` builds
 * its own. Throws when the provider found no credential, or when no adapter
 * speaks the model's `api`.
 */
export function createModelAdapter(
	provider: Provider,
	model: FlueModel,
	options: ModelAdapterOptions,
) {
	if (provider.createAdapter) return provider.createAdapter(model, options);
	const { auth } = options;
	if (!auth) throw new Error(`No API key for provider: ${model.provider}`);
	const request = {
		model,
		auth: auth.auth,
		baseURL: withEnv(auth.auth.baseUrl ?? model.baseUrl, auth.env),
		headers: requestHeaders(provider, model, auth, options.promptCacheKey),
		...(options.fetch ? { fetch: options.fetch } : {}),
	};
	const { api } = model;
	switch (api) {
		case 'anthropic-messages':
			return anthropicText(model.id, {
				...anthropicCredential(request),
				baseURL: request.baseURL,
				defaultHeaders: request.headers,
				...(options.fetch ? { fetch: options.fetch } : {}),
				oauth: auth.auth.apiKey?.includes('sk-ant-oat') === true,
				allowEmptySignature: model.compat?.allowEmptySignature === true,
				provider: model.provider,
				reasoning: reasoningFor(model, 'anthropic-messages'),
				midConversationChannels: anthropicChannels(model.provider),
			});
		case 'openai-responses':
			if (model.provider !== 'openai') return compatibleAdapter(model, request, 'responses');
			return createOpenaiChat(model.id, clientKey(request), {
				baseURL: request.baseURL,
				defaultHeaders: request.headers,
				...(options.fetch ? { fetch: options.fetch } : {}),
				reasoning: reasoningFor(model, 'openai-responses'),
				// A `baseURL` turns the channels off; this is OpenAI's own API.
				midConversationChannels: true,
			});
		case 'azure-openai-responses':
			return azureOpenaiText(model.id, azureConfig(model, request, auth));
		case 'openai-completions':
			return compatibleAdapter(model, request, 'chat-completions');
		case 'google-generative-ai':
			return createGeminiChat(model.id, clientKey(request), {
				baseURL: request.baseURL,
				// The catalog URL has the version path, so the SDK must not add one.
				httpOptions: { apiVersion: '' },
				defaultHeaders: withoutNulls(request.headers),
				reasoning: reasoningFor(model, 'google-generative-ai'),
			});
		case 'google-vertex':
			return createGeminiChat(model.id, auth.auth.apiKey ?? '', vertexConfig(model, request, auth));
		case 'bedrock-converse-stream':
			return bedrockAdapter(model, auth);
		case 'mistral-conversations':
			return createMistralText(model.id, clientKey(request), {
				baseURL: request.baseURL,
				defaultHeaders: withoutNulls(request.headers),
				reasoning: reasoningFor(model, 'mistral-conversations'),
			});
		default:
			throw new Error(
				`[flue] Model "${model.provider}/${model.id}" uses the "${api}" API, which no ` +
					'adapter speaks. Give its provider a createAdapter function.',
			);
	}
}

interface AdapterRequest {
	model: FlueModel;
	auth: ModelAuth;
	baseURL: string;
	headers: Record<string, string | null>;
	fetch?: typeof fetch;
}

/** The header that carries the credential, when the auth puts it there. */
function credentialHeader(headers: AdapterRequest['headers']) {
	const names = ['authorization', 'x-api-key', 'cf-aig-authorization'];
	return Object.entries(headers).find(
		([name, value]) => names.includes(name.toLowerCase()) && value !== null,
	);
}

/**
 * pi's rule for the SDK key: the auth's key, or a placeholder when a header
 * carries the credential (the header replaces the SDK's own).
 */
function clientKey(request: AdapterRequest) {
	if (request.auth.apiKey) return request.auth.apiKey;
	if (credentialHeader(request.headers)) return 'unused';
	throw new Error(`No API key for provider: ${request.model.provider}`);
}

/** A bearer header is an auth token for the Anthropic SDK; anything else follows `clientKey`. */
function anthropicCredential(request: AdapterRequest) {
	const [name, value] = credentialHeader(request.headers) ?? [];
	const bearer =
		name?.toLowerCase() === 'authorization' ? value?.match(/^Bearer (.+)$/)?.[1] : undefined;
	if (!request.auth.apiKey && bearer) return { authToken: bearer };
	return { apiKey: clientKey(request) };
}

/** Chat Completions, and Responses on a host that is not OpenAI: the record's quirks drive the request. */
function compatibleAdapter(
	model: FlueModel,
	request: AdapterRequest,
	wire: 'chat-completions' | 'responses',
) {
	return openaiCompatibleText(model.id, {
		name: model.provider,
		api: wire,
		baseURL: request.baseURL,
		apiKey: clientKey(request),
		defaultHeaders: request.headers,
		...(request.fetch ? { fetch: request.fetch } : {}),
		...(model.compat ? { compat: model.compat } : {}),
		reasoning: model.reasoning && (model.reasoningMap ?? true),
	});
}

function azureConfig(model: FlueModel, request: AdapterRequest, auth: AuthResult) {
	const env = auth.env ?? {};
	const resourceName = env.AZURE_OPENAI_RESOURCE_NAME;
	// pi's order: the base URL variable, then the resource name, then the record.
	const baseURL = env.AZURE_OPENAI_BASE_URL ?? (resourceName ? undefined : request.baseURL);
	return {
		apiKey: clientKey(request),
		...(baseURL ? { baseURL } : {}),
		...(resourceName ? { resourceName } : {}),
		...(env.AZURE_OPENAI_API_VERSION ? { apiVersion: env.AZURE_OPENAI_API_VERSION } : {}),
		deploymentNameMap: parseDeploymentNameMap(env.AZURE_OPENAI_DEPLOYMENT_NAME_MAP),
		defaultHeaders: request.headers,
		...(request.fetch ? { fetch: request.fetch } : {}),
		reasoning: reasoningFor(model, 'azure-openai-responses'),
	};
}

/** pi's format: `model=deployment` pairs, separated by commas. */
function parseDeploymentNameMap(value: string | undefined) {
	const map: Record<string, string> = {};
	for (const entry of value?.split(',') ?? []) {
		const [modelId, deployment] = entry.trim().split('=', 2);
		if (modelId?.trim() && deployment?.trim()) map[modelId.trim()] = deployment.trim();
	}
	return map;
}

/** An API key (express mode), or Application Default Credentials with a project and a location. */
function vertexConfig(model: FlueModel, request: AdapterRequest, auth: AuthResult) {
	const env = auth.env ?? {};
	// The catalog URL has a `{location}` template: the SDK builds that endpoint itself.
	const customBaseURL = request.baseURL.includes('{location}')
		? undefined
		: request.baseURL || undefined;
	return {
		vertexai: true,
		apiVersion: 'v1',
		...(request.auth.apiKey
			? {}
			: {
					project: env.GOOGLE_CLOUD_PROJECT,
					location: env.GOOGLE_CLOUD_LOCATION,
				}),
		...(customBaseURL ? { baseURL: customBaseURL } : {}),
		defaultHeaders: withoutNulls(request.headers),
		reasoning: reasoningFor(model, 'google-vertex'),
	};
}

// ponytail: Bedrock takes the region from the model ARN, then AWS_REGION and
// AWS_DEFAULT_REGION, then us-east-1. pi also let an AWS_PROFILE pick the
// region; add it when a user needs it.
function bedrockAdapter(model: FlueModel, auth: AuthResult) {
	const env = auth.env ?? {};
	const bearerToken = env.AWS_BEARER_TOKEN_BEDROCK;
	const arnRegion = model.id.match(/^arn:aws(?:-[a-z0-9-]+)?:bedrock:([a-z0-9-]+):/)?.[1];
	return createBedrockConverse(model.id, bearerToken ?? '', {
		region: arnRegion ?? env.AWS_REGION ?? env.AWS_DEFAULT_REGION ?? 'us-east-1',
		auth: bearerToken ? 'apikey' : 'sigv4',
		// A custom endpoint from the auth, for example a VPC endpoint or a proxy.
		...(auth.auth.baseUrl ? { baseURL: auth.auth.baseUrl } : {}),
		reasoning: reasoningFor(model, 'bedrock-converse-stream'),
	});
}

/** The record's reasoning data. `api` matters on the Anthropic wire, where it adds the thinking shape. */
function reasoningFor(model: FlueModel, api: WireApi) {
	return modelReasoning({ ...model, api });
}

/**
 * pi's mid-conversation flags: system prompts and tool changes for Anthropic's
 * own API, and only system prompts for OpenCode. The adapter's table limits
 * both to the models that have them.
 */
function anthropicChannels(providerId: string) {
	if (providerId === 'anthropic') return true;
	if (providerId === 'opencode') return { systemPrompts: true };
	return false;
}

/**
 * The provider's, then the record's, then the auth's headers. A `null` value
 * removes the header. pi's session headers go first, so the others can
 * replace them: the affinity header of a model whose compat asks for it, and
 * OpenCode's conversation header.
 */
function requestHeaders(provider: Provider, model: FlueModel, auth: AuthResult, sessionId: string) {
	const headers: Record<string, string | null> = {
		...sessionHeaders(model, sessionId),
		...provider.headers,
		...model.headers,
		...auth.auth.headers,
	};
	const isOpenCode = model.provider === 'opencode' || model.provider === 'opencode-go';
	const hasSessionHeader = Object.keys(headers).some(
		(name) => name.toLowerCase() === 'x-opencode-session',
	);
	if (isOpenCode && !hasSessionHeader) headers['x-opencode-session'] = sessionId;
	return headers;
}

/** pi sends these on the Anthropic wire. The OpenAI-compatible adapter sends its own. */
function sessionHeaders(model: FlueModel, sessionId: string) {
	if (model.api !== 'anthropic-messages') return {};
	const isOpenRouter = model.provider === 'openrouter' || model.baseUrl.includes('openrouter.ai');
	const sendsAffinity = model.compat?.sendSessionAffinityHeaders ?? isOpenRouter;
	if (!sendsAffinity) return {};
	const format = model.compat?.sessionAffinityFormat ?? (isOpenRouter ? 'openrouter' : undefined);
	const name = format === 'openrouter' ? 'x-session-id' : 'x-session-affinity';
	return { [name]: sessionId };
}

/** For the SDKs that cannot remove a header. */
function withoutNulls(headers: Record<string, string | null>) {
	const kept: Record<string, string> = {};
	for (const [name, value] of Object.entries(headers)) {
		if (value !== null) kept[name] = value;
	}
	return kept;
}

/** Fill `{NAME}` from the auth's config, for example a Cloudflare account id. */
function withEnv(url: string, env: AuthResult['env']) {
	return url.replace(/\{([A-Z0-9_]+)\}/g, (match, name: string) => env?.[name] ?? match);
}

/** The `modelOptions` key that caps the output tokens, per wire. */
const MAX_TOKENS_KEY: Partial<Record<string, string>> = {
	'anthropic-messages': 'max_tokens',
	'openai-responses': 'max_output_tokens',
	'azure-openai-responses': 'max_output_tokens',
	'openai-completions': 'max_completion_tokens',
	'google-generative-ai': 'maxOutputTokens',
	'google-vertex': 'maxOutputTokens',
	'bedrock-converse-stream': 'max_completion_tokens',
	'mistral-conversations': 'max_tokens',
};

/** The `modelOptions` that cap the output of a call on the model's wire. */
export function outputTokenOptions(model: Pick<FlueModel, 'api'>, maxTokens: number) {
	const key = MAX_TOKENS_KEY[model.api];
	return key ? { [key]: maxTokens } : {};
}

const CONTEXT_SAFETY_TOKENS = 4096;

/** pi's output cap: the model's `maxTokens`, cut to what the context leaves free. */
export function clampMaxTokensToContext(
	model: Pick<FlueModel, 'contextWindow' | 'maxTokens'>,
	contextTokens: number,
) {
	if (model.contextWindow <= 0) return Math.max(1, model.maxTokens);
	const available = model.contextWindow - contextTokens - CONTEXT_SAFETY_TOKENS;
	return Math.min(model.maxTokens, Math.max(1, available));
}

export interface ModelCallHooks {
	/** Runs each step of the model stream, for the execution interceptor. */
	intercept<T>(run: () => Promise<T>): Promise<T>;
	/** The tokens that the request's context takes, for the output cap. */
	contextTokens(options: TextOptions): number;
}

/**
 * `adapter` with Flue's per-request defaults (the output cap and, on
 * Responses, `store: false`), and with each step of its stream run through
 * `hooks.intercept`. A caller's own `modelOptions` win.
 */
export function wrapModelAdapter(adapter: AnyTextAdapter, model: FlueModel, hooks: ModelCallHooks) {
	const isResponses = model.api === 'openai-responses' || model.api === 'azure-openai-responses';
	const chatStream: AnyTextAdapter['chatStream'] = (options) => {
		const defaults = {
			...outputTokenOptions(model, clampMaxTokensToContext(model, hooks.contextTokens(options))),
			...(isResponses ? { store: false } : {}),
		};
		const stream = adapter.chatStream({
			...options,
			modelOptions: { ...defaults, ...options.modelOptions },
		});
		return intercepted(stream, hooks);
	};
	// The adapter's own methods stay on the prototype.
	const wrapped: AnyTextAdapter = Object.create(adapter);
	return Object.assign(wrapped, { chatStream });
}

async function* intercepted<T>(stream: AsyncIterable<T>, hooks: ModelCallHooks) {
	const iterator = stream[Symbol.asyncIterator]();
	let finished = false;
	try {
		while (true) {
			const step = await hooks.intercept(() => iterator.next());
			if (step.done) {
				finished = true;
				return;
			}
			yield step.value;
		}
	} finally {
		const close = iterator.return?.bind(iterator);
		if (!finished && close) await hooks.intercept(() => close());
	}
}
