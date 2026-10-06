/**
 * The `cloudflare` provider: models dispatched through the Workers AI binding
 * (`env.AI.run()`) instead of HTTP, on the TanStack AI adapters.
 *
 * Wire format: `anthropic/…` AI Gateway models speak Anthropic Messages and
 * `openai/…` models speak OpenAI Responses, through `cloudflareBindingFetch`.
 * Everything else — Workers AI `@cf/…` ids and other gateway vendors — speaks
 * OpenAI-compatible Chat Completions, with the Workers AI quirks merged under
 * each model's own catalog compat. Catalogued gateway models dispatch by
 * their catalog `api`; ids no catalog knows fall back to their vendor prefix.
 */
import { type AnyTextAdapter, EventType, type StreamChunk } from '@tanstack/ai';
import { createAnthropicChat } from '@tanstack/ai-anthropic';
import { cloudflareBindingFetch } from '@tanstack/ai-cloudflare';
import {
	getModels,
	type ModelCompat,
	modelReasoning,
	type ReasoningMap,
} from '@tanstack/ai-models';
import { createOpenaiChat } from '@tanstack/ai-openai';
import { openaiCompatibleText } from '@tanstack/ai-openai/compatible';
import { CloudflareAIBindingError, RETRYABLE_INTERRUPTION_MARKER } from '../errors.ts';
import { createProvider, type FlueModel, type ModelAdapterOptions } from '../providers/provider.ts';
import { withAnthropicGatewayIds } from '../providers/registry.ts';
import type { CloudflareGatewayOptions } from './gateway.ts';

/**
 * The `api` marker carried by Workers AI catalog models and zero-metadata
 * dynamic ids. `bindingWireFormat` reads it, alongside the real gateway apis,
 * to pick a serialization.
 */
const CLOUDFLARE_AI_BINDING_API = 'cloudflare-ai-binding';

/**
 * The Workers AI quirks on the Chat Completions wire (pi's
 * `detectCompat('cloudflare-workers-ai')`). Each model's catalog compat wins,
 * field by field. The binding sends `x-session-affinity` itself, so the
 * adapter's own affinity headers stay off.
 */
const WORKERS_AI_COMPAT: ModelCompat = {
	supportsStore: false,
	supportsDeveloperRole: false,
	supportsReasoningEffort: true,
	supportsUsageInStreaming: true,
	maxTokensField: 'max_completion_tokens',
	requiresReasoningContentOnAssistantMessages: false,
	thinkingFormat: 'openai',
	zaiToolStream: false,
	supportsStrictMode: true,
	supportsOpenAIGrammarTools: false,
	sendSessionAffinityHeaders: false,
	supportsLongCacheRetention: false,
};

/** The binding's accepted efforts: `minimal` reads as `low`, and the top levels as `high`. */
const WORKERS_AI_EFFORTS: ReasoningMap = {
	minimal: 'low',
	low: 'low',
	medium: 'medium',
	high: 'high',
	xhigh: 'high',
	max: 'high',
};

/**
 * Cap on how long a model stream may go without delivering a byte before the
 * request fails as a retryable interruption. Generous on purpose: a
 * long-thinking model can be silent for minutes when neither keepalives nor
 * reasoning deltas stream, and a false trip burns a turn retry. The case this
 * exists for — a stream that returned 200 and then never speaks again — is
 * unbounded without it.
 */
const DEFAULT_STREAM_IDLE_TIMEOUT_MS = 300_000;

/**
 * Wrap a response body so a chunk gap longer than `idleMs` rejects the read
 * with a retryable-interruption error instead of pending forever. The timer
 * only runs while a read is outstanding — consumer backpressure is not
 * source silence. `idleMs <= 0` disables the guard.
 */
function withStreamIdleDeadline(
	body: ReadableStream<Uint8Array>,
	idleMs: number,
): ReadableStream<Uint8Array> {
	if (idleMs <= 0) return body;
	const reader = body.getReader();
	return new ReadableStream<Uint8Array>({
		async pull(controller) {
			let timer: ReturnType<typeof setTimeout> | undefined;
			try {
				const result = await Promise.race([
					reader.read(),
					new Promise<never>((_, reject) => {
						timer = setTimeout(() => {
							reject(
								new Error(
									`Model stream stalled: no data received for ${Math.round(idleMs / 1000)}s ${RETRYABLE_INTERRUPTION_MARKER}`,
								),
							);
						}, idleMs);
					}),
				]);
				if (result.done) controller.close();
				else controller.enqueue(result.value);
			} catch (error) {
				// Cancel the source so workerd doesn't keep the underlying AI request
				// streaming with no consumer. cancel() rejects when the source errored
				// in the meantime — consume it: an unhandled rejection is an
				// exception on workerd.
				try {
					void reader.cancel().catch(() => {});
				} catch {}
				controller.error(error);
			} finally {
				clearTimeout(timer);
			}
		},
		cancel(reason) {
			return reader.cancel(reason);
		},
	});
}

/**
 * Workers AI streams end with a usage-only trailer that has no `choices`;
 * the Chat Completions reader indexes `choices[0]` on every event, so give
 * such events an empty list (OpenAI's own trailer shape).
 */
function withChoices(body: ReadableStream<Uint8Array>) {
	const encoder = new TextEncoder();
	let buffer = '';
	const fix = (line: string) => {
		if (!line.startsWith('data:')) return line;
		const data = line.slice(5).trim();
		if (data === '' || data === '[DONE]') return line;
		try {
			const event: unknown = JSON.parse(data);
			if (typeof event !== 'object' || event === null || 'choices' in event) return line;
			return `data: ${JSON.stringify({ ...event, choices: [] })}`;
		} catch {
			return line;
		}
	};
	return body.pipeThrough(new TextDecoderStream()).pipeThrough(
		new TransformStream<string, Uint8Array>({
			transform(text, controller) {
				buffer += text;
				const lines = buffer.split('\n');
				buffer = lines.pop() ?? '';
				for (const line of lines) controller.enqueue(encoder.encode(`${fix(line)}\n`));
			},
			flush(controller) {
				if (buffer) controller.enqueue(encoder.encode(fix(buffer)));
			},
		}),
	);
}

/**
 * The gateway shape forwarded to `ai.run`, plus the header that carries the
 * one option with no binding-object equivalent (`requestTimeoutMs` →
 * `cf-aig-request-timeout`).
 */
function gatewayRunOptions(gateway: CloudflareGatewayOptions | undefined) {
	if (!gateway) return { gateway: undefined, headers: {} };
	const { requestTimeoutMs, ...forwarded } = gateway;
	return {
		gateway: forwarded,
		headers:
			requestTimeoutMs !== undefined && requestTimeoutMs > 0
				? { 'cf-aig-request-timeout': String(requestTimeoutMs) }
				: {},
	};
}

async function safeReadText(response: Response) {
	try {
		return await response.text();
	} catch {
		return undefined;
	}
}

/** What one binding call saw: the gateway log id of the last response. */
interface BindingCall {
	gatewayLogId?: string;
}

/**
 * A binding fetch with Flue's response rules: the response's own gateway log
 * id is kept, a failure reads as `CloudflareAIBindingError` (with the
 * context-overflow marker on a 413), and the body gets the idle deadline.
 */
function guardedFetch(run: typeof fetch, idleMs: number, call: BindingCall): typeof fetch {
	return async (input, init) => {
		const response = await run(input, init);
		// This response's OWN header — never env.AI.aiGatewayLogId, which
		// reflects the binding's most recent request and cross-attributes
		// under concurrency.
		const gatewayLogId = response.headers.get('cf-aig-log-id');
		if (gatewayLogId) call.gatewayLogId = gatewayLogId;
		if (!response.ok) {
			const error = new CloudflareAIBindingError({
				status: response.status,
				statusText: response.statusText,
				body: await safeReadText(response),
			});
			return Response.json(
				{ error: { message: error.message } },
				{ status: response.status, statusText: response.statusText },
			);
		}
		if (!response.body) return response;
		return new Response(withStreamIdleDeadline(response.body, idleMs), response);
	};
}

/** `env.AI` as the Chat Completions endpoint of a Workers AI model. */
function workersAiFetch(
	ai: CloudflareAIBinding,
	gateway: CloudflareGatewayOptions | undefined,
	sessionId: string,
): typeof fetch {
	const run = gatewayRunOptions(gateway);
	return async (_input, init) => {
		const { model, ...body } = JSON.parse(typeof init?.body === 'string' ? init.body : '{}');
		const result = await ai.run(String(model ?? ''), body, {
			returnRawResponse: true,
			...(init?.signal ? { signal: init.signal } : {}),
			extraHeaders: { 'x-session-affinity': sessionId, ...run.headers },
			...(run.gateway ? { gateway: run.gateway } : {}),
		});
		if (!(result instanceof Response)) return Response.json(result);
		if (!result.ok || !result.body) return result;
		return new Response(withChoices(result.body), result);
	};
}

/**
 * `adapter` with the binding call's response metadata on each run's end:
 * the gateway log id and the provider's own finish value, for telemetry.
 */
function withResponseDiagnostics(adapter: AnyTextAdapter, call: BindingCall) {
	const chatStream: AnyTextAdapter['chatStream'] = async function* (options) {
		for await (const chunk of adapter.chatStream(options)) {
			const isEnd = chunk.type === EventType.RUN_FINISHED || chunk.type === EventType.RUN_ERROR;
			yield isEnd ? withDiagnostics(chunk, call) : chunk;
		}
	};
	const wrapped: AnyTextAdapter = Object.create(adapter);
	return Object.assign(wrapped, { chatStream });
}

function withDiagnostics(chunk: StreamChunk, call: BindingCall): StreamChunk {
	const providerFinishReason =
		chunk.type === EventType.RUN_FINISHED ? chunk.finishReason : undefined;
	const providerResponse = {
		...(call.gatewayLogId ? { gatewayLogId: call.gatewayLogId } : {}),
		...(providerFinishReason ? { providerFinishReason } : {}),
	};
	if (Object.keys(providerResponse).length === 0) return chunk;
	return {
		...chunk,
		metadata: { ...chunk.metadata, flue: { providerResponse } },
	};
}

/** The binding type that `cloudflareBindingFetch` takes. */
type TanStackBinding = Parameters<typeof cloudflareBindingFetch>[0]['binding'];

type BindingWireFormat = 'anthropic-messages' | 'openai-completions' | 'openai-responses';

/**
 * The serialization a binding model speaks. Catalogued gateway models
 * dispatch by their catalog `api`; ids no catalog knows fall back to their
 * gateway vendor prefix. Everything else — `@cf/…` ids and unknown gateway
 * vendors — speaks OpenAI-compatible Chat Completions. An api with no wire
 * format here yields `undefined`, so the caller errors instead of sending
 * the wrong shape.
 */
function bindingWireFormat(model: FlueModel): BindingWireFormat | undefined {
	if (model.api === 'anthropic-messages' || model.id.startsWith('anthropic/'))
		return 'anthropic-messages';
	if (model.api === 'openai-responses' || model.id.startsWith('openai/')) return 'openai-responses';
	if (model.api === 'openai-completions' || model.api === CLOUDFLARE_AI_BINDING_API)
		return 'openai-completions';
	return undefined;
}

/** The gateway model id without its vendor prefix: the binding fetch adds it back. */
function vendorModelId(modelId: string, vendor: string) {
	const prefix = `${vendor}/`;
	return modelId.startsWith(prefix) ? modelId.slice(prefix.length) : modelId;
}

interface BindingSettings {
	ai: CloudflareAIBinding;
	gateway: CloudflareGatewayOptions | undefined;
	streamIdleTimeoutMs: number;
	cacheRetention: CloudflareCacheRetention;
}

/** The adapter for one call of `model` through the binding. */
function bindingAdapter(
	model: FlueModel,
	settings: BindingSettings,
	options: ModelAdapterOptions,
): AnyTextAdapter {
	const call: BindingCall = {};
	const run = gatewayRunOptions(settings.gateway);
	const headers = {
		'x-session-affinity': options.promptCacheKey,
		...run.headers,
	};
	// The binding is typed structurally (so Node can import this module), and
	// `cloudflareBindingFetch` only calls its `run()`: no wider type check fits.
	const binding = settings.ai as unknown as TanStackBinding;
	switch (bindingWireFormat(model)) {
		case 'anthropic-messages': {
			const adapter = createAnthropicChat(
				vendorModelId(model.id, 'anthropic'),
				'cloudflare-binding',
				{
					fetch: guardedFetch(
						cloudflareBindingFetch({
							binding,
							vendor: 'anthropic',
							...(run.gateway ? { gateway: run.gateway } : {}),
						}),
						settings.streamIdleTimeoutMs,
						call,
					),
					defaultHeaders: headers,
					// Flue's turn retries own transient errors; the SDK must not retry too.
					maxRetries: 0,
					provider: 'cloudflare',
					reasoning: modelReasoning({ ...model, api: 'anthropic-messages' }),
				},
			);
			return withResponseDiagnostics(
				withPromptCacheRetention(adapter, settings.cacheRetention),
				call,
			);
		}
		case 'openai-responses':
			return withResponseDiagnostics(
				createOpenaiChat(vendorModelId(model.id, 'openai'), 'cloudflare-binding', {
					fetch: guardedFetch(
						cloudflareBindingFetch({
							binding,
							vendor: 'openai',
							...(run.gateway ? { gateway: run.gateway } : {}),
						}),
						settings.streamIdleTimeoutMs,
						call,
					),
					defaultHeaders: headers,
					maxRetries: 0,
					reasoning: modelReasoning({ ...model, api: 'openai-responses' }),
					// pi sent `additional_tools` through the binding when the model has the channel.
					midConversationChannels: model.compat?.supportsAdditionalTools === true,
				}),
				call,
			);
		case 'openai-completions':
			return withResponseDiagnostics(
				openaiCompatibleText(model.id, {
					name: 'cloudflare',
					api: 'chat-completions',
					baseURL: 'https://workers-ai.binding.invalid/v1',
					apiKey: 'cloudflare-binding',
					fetch: guardedFetch(
						workersAiFetch(settings.ai, settings.gateway, options.promptCacheKey),
						settings.streamIdleTimeoutMs,
						call,
					),
					maxRetries: 0,
					compat: { ...WORKERS_AI_COMPAT, ...model.compat },
					reasoning: model.reasoning && (model.reasoningMap ?? WORKERS_AI_EFFORTS),
				}),
				call,
			);
		default:
			throw new Error(
				`Cloudflare AI binding has no wire format for api "${model.api}" (model "${model.id}").`,
			);
	}
}

/** The binding's Anthropic path caches by the provider's own retention setting. */
function withPromptCacheRetention(adapter: AnyTextAdapter, retention: CloudflareCacheRetention) {
	const chatStream: AnyTextAdapter['chatStream'] = (options) =>
		adapter.chatStream({
			...options,
			promptCache: { ...options.promptCache, retention },
		});
	const wrapped: AnyTextAdapter = Object.create(adapter);
	return Object.assign(wrapped, { chatStream });
}

// ─── Provider factory ───────────────────────────────────────────────────────

/**
 * Minimal Workers AI binding shape. Kept structural so the factory type stays
 * importable on Node.
 */
export interface CloudflareAIBinding {
	run(
		modelId: string,
		inputs: Record<string, unknown>,
		options?: Record<string, unknown>,
	): Promise<Response | Record<string, unknown>>;
}

/**
 * Anthropic prompt-cache retention for the binding's Anthropic path
 * (`anthropic/…` gateway models). The Workers AI binding forwards
 * `cache_control` on message blocks and tools to Anthropic, so opt-in caching
 * serves repeated prefixes at the cached input rate instead of full price.
 * `'long'` requests the 1-hour TTL where the platform supports it. Default
 * `'none'` keeps the current behavior (no cache markers).
 */
export type CloudflareCacheRetention = 'none' | 'short' | 'long';

export interface CloudflareBindingProviderOptions {
	/** The captured `env.AI` reference. */
	binding: CloudflareAIBinding;
	/**
	 * AI Gateway options forwarded to every `env.AI.run(...)` call routed
	 * through this provider.
	 *
	 * - Omitted: routes through Cloudflare's default AI Gateway, which the
	 *   binding spins up on demand for the account.
	 * - Options object: replaces the default. Specify `id` plus any other
	 *   knobs (cache, metadata, logging).
	 * - `false`: opts out — no gateway is passed to `ai.run`.
	 *
	 * See https://developers.cloudflare.com/ai-gateway/integrations/worker-binding-methods/.
	 */
	gateway?: CloudflareGatewayOptions | false;
	/**
	 * Cap on how long a model stream may go without delivering a byte before
	 * the request fails as a retryable interruption (the turn retries under
	 * the transient-error budget). Defaults to 5 minutes — generous, because
	 * a long-thinking model can be legitimately silent when neither
	 * keepalives nor reasoning deltas stream. `0` disables the guard.
	 */
	streamIdleTimeoutMs?: number;
	/**
	 * Anthropic prompt-cache retention for `anthropic/…` models routed
	 * through the binding. Default `'none'` matches the current behavior (no
	 * `cache_control` markers); opt in with `'short'` (5-minute TTL) or
	 * `'long'` (1-hour TTL where supported) to cache repeated prefixes and
	 * pay the cached input rate on cache hits. The conversation's prompt
	 * cache key keeps the cache reusable across turns.
	 */
	cacheRetention?: CloudflareCacheRetention;
}

/**
 * The `cloudflare` provider: models dispatched through the Workers AI binding
 * (`env.AI.run()`) instead of HTTP. Model metadata comes from the
 * `cloudflare-workers-ai` catalog (`@cf/…` ids) and the
 * `cloudflare-ai-gateway` catalog (`anthropic/…` and `openai/…` ids); ids
 * neither catalog knows resolve with zero metadata, since the binding accepts
 * arbitrary model ids.
 *
 * The generated worker entry registers it when the `providers` config is
 * omitted or lists `'cloudflare'`; call `setProvider()` with this factory in
 * `app.ts` to override the gateway options (a user registration wins over
 * the generated one).
 */
export function cloudflareBindingProvider(options: CloudflareBindingProviderOptions) {
	// The documented tri-state: omitted routes through Cloudflare's default AI
	// Gateway, `false` opts out, an options object replaces the default.
	const settings: BindingSettings = {
		ai: options.binding,
		gateway: options.gateway === false ? undefined : (options.gateway ?? { id: 'default' }),
		streamIdleTimeoutMs: options.streamIdleTimeoutMs ?? DEFAULT_STREAM_IDLE_TIMEOUT_MS,
		cacheRetention: options.cacheRetention ?? 'none',
	};
	return createProvider({
		id: 'cloudflare',
		name: 'Cloudflare Workers AI',
		// Keyless: the binding itself is the credential.
		auth: {
			apiKey: {
				name: 'Cloudflare AI binding',
				resolve: async () => ({ auth: {} }),
			},
		},
		models: [...bindingCatalogModels(), ...gatewayCatalogModels()],
		dynamicModels: { api: CLOUDFLARE_AI_BINDING_API, baseUrl: '' },
		createAdapter: (model, adapterOptions) => bindingAdapter(model, settings, adapterOptions),
	});
}

/**
 * The `cloudflare-workers-ai` catalog re-tagged for the binding: same ids and
 * metadata (each model's compat included), dispatched through this provider
 * instead of the REST API.
 */
function bindingCatalogModels(): FlueModel[] {
	return getModels('cloudflare-workers-ai').map((model) => ({
		...model,
		api: CLOUDFLARE_AI_BINDING_API,
		provider: 'cloudflare',
		baseUrl: '',
	}));
}

/**
 * The `cloudflare-ai-gateway` catalog re-tagged for the binding. A gateway
 * catalog id is bare (`gpt-5.6-terra`) and names its vendor in the entry's
 * gateway-URL path segment (`…/{CLOUDFLARE_GATEWAY_ID}/openai`); the binding
 * addresses the same model as `openai/gpt-5.6-terra`. Anthropic ids are
 * dashed, as Anthropic's API takes them. Models keep the catalog's `api`,
 * capabilities, and cost data. `/compat` entries are skipped: they alias
 * `@cf/…` ids the Workers AI catalog already declares.
 */
function gatewayCatalogModels(): FlueModel[] {
	return withAnthropicGatewayIds(getModels('cloudflare-ai-gateway')).flatMap((model) => {
		const vendor = model.baseUrl.slice(model.baseUrl.lastIndexOf('/') + 1);
		if (vendor.length === 0 || vendor === 'compat') return [];
		return [
			{
				...model,
				id: `${vendor}/${model.id}`,
				provider: 'cloudflare',
				baseUrl: '',
			},
		];
	});
}
