import {
	createServer,
	type IncomingHttpHeaders,
	type ServerHttp2Session,
	type ServerHttp2Stream,
} from 'node:http2';
import { type AnyTextAdapter, chat, type PromptCacheOptions } from '@tanstack/ai';
import { getModels, getProviders } from '@tanstack/ai-models';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { clampMaxTokensToContext, createModelAdapter, wrapModelAdapter } from './adapters.ts';
import { builtinProvider } from './builtins.ts';
import type { AuthResult, FlueModel } from './provider.ts';

interface SentRequest {
	url: string;
	headers: Headers;
	body: Record<string, unknown>;
}

const keyAuth: AuthResult = { auth: { apiKey: 'test-key' } };

afterEach(() => {
	vi.unstubAllGlobals();
});

function builtin(providerId: string, modelId: string) {
	const record = getProviders().find((provider) => provider.id === providerId);
	if (!record) throw new Error(`No provider "${providerId}" in the catalog.`);
	const provider = builtinProvider({
		provider: record,
		models: getModels(providerId),
	});
	const model = provider.getModels().find((entry) => entry.id === modelId);
	if (!model) throw new Error(`No model "${providerId}/${modelId}".`);
	return { provider, model };
}

/** Flue's adapter for the model, as a session builds it, with an empty context. */
function flueAdapter(providerId: string, modelId: string, auth: AuthResult) {
	const { provider, model } = builtin(providerId, modelId);
	const adapter = createModelAdapter(provider, model, {
		auth,
		promptCacheKey: 'conv-1',
	});
	return wrapModelAdapter(adapter, model, {
		intercept: (run) => run(),
		contextTokens: () => 0,
	});
}

/** One turn. The test servers answer with an error, so the turn ends after the request. */
async function runTurn(
	adapter: AnyTextAdapter,
	promptCache: PromptCacheOptions = { key: 'conv-1' },
) {
	try {
		for await (const _chunk of chat({
			adapter,
			messages: [{ role: 'user', content: 'Hi' }],
			promptCache,
		})) {
			// Drain the stream.
		}
	} catch {
		// The stub answers 400.
	}
}

async function toSentRequest(request: Request) {
	return {
		url: request.url,
		headers: request.headers,
		body: JSON.parse(await request.text()),
	};
}

/** The first HTTP request of one turn, read at the `fetch` boundary. */
async function firstRequest(
	providerId: string,
	modelId: string,
	auth: AuthResult = keyAuth,
	promptCache?: PromptCacheOptions,
) {
	const sent: SentRequest[] = [];
	vi.stubGlobal('fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
		sent.push(await toSentRequest(new Request(input, init)));
		return Response.json({ error: { message: 'test stop' } }, { status: 400 });
	});
	await runTurn(flueAdapter(providerId, modelId, auth), promptCache);
	const [request] = sent;
	if (!request) throw new Error('The adapter sent no request.');
	return request;
}

/**
 * A local HTTP/2 server for the AWS SDK, which does not use `fetch`. It
 * records each request and answers 400.
 */
async function http2TestServer() {
	const sent: SentRequest[] = [];
	const sessions = new Set<ServerHttp2Session>();
	const server = createServer();
	server.on('session', (session) => sessions.add(session));
	server.on('stream', (stream: ServerHttp2Stream, headers: IncomingHttpHeaders) => {
		const chunks: Buffer[] = [];
		stream.on('data', (chunk: Buffer) => chunks.push(chunk));
		stream.on('end', () => {
			sent.push({
				url: String(headers[':path']),
				headers: new Headers(
					Object.entries(headers).flatMap(([name, value]) =>
						typeof value === 'string' && !name.startsWith(':') ? [[name, value]] : [],
					),
				),
				body: JSON.parse(Buffer.concat(chunks).toString('utf8')),
			});
			stream.respond({ ':status': 400, 'content-type': 'application/json' });
			stream.end(JSON.stringify({ message: 'test stop' }));
		});
	});
	await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
	const address = server.address();
	if (!address || typeof address === 'string') throw new Error('The test server has no port.');
	return {
		sent,
		url: `http://127.0.0.1:${address.port}`,
		close: () =>
			new Promise((resolve) => {
				for (const session of sessions) session.destroy();
				server.close(resolve);
			}),
	};
}

describe('createModelAdapter requests', () => {
	it('anthropic-messages: the key, the record id, and the output cap', async () => {
		const request = await firstRequest('anthropic', 'claude-sonnet-4-5');

		// TanStack sends every Messages request to the SDK's beta endpoint.
		expect(request.url).toBe('https://api.anthropic.com/v1/messages?beta=true');
		expect(request.headers.get('x-api-key')).toBe('test-key');
		expect(request.body).toMatchObject({
			model: 'claude-sonnet-4-5',
			max_tokens: 64_000,
			stream: true,
		});
		expect(JSON.stringify(request.body)).toContain('cache_control');
	});

	it('anthropic-messages with promptCache "none" sends no cache markers', async () => {
		const request = await firstRequest('anthropic', 'claude-sonnet-4-5', keyAuth, {
			retention: 'none',
			key: 'conv-1',
		});

		expect(JSON.stringify(request.body)).not.toContain('cache_control');
	});

	it('anthropic-messages sends ANTHROPIC_AUTH_TOKEN as a bearer header and no key', async () => {
		const request = await firstRequest('anthropic', 'claude-sonnet-4-5', {
			auth: { headers: { Authorization: 'Bearer auth-token' } },
		});

		expect(request.headers.get('authorization')).toBe('Bearer auth-token');
		expect(request.headers.get('x-api-key')).toBeNull();
	});

	it('openai-responses: store false and the conversation prompt cache key', async () => {
		const request = await firstRequest('openai', 'gpt-5');

		expect(request.url).toBe('https://api.openai.com/v1/responses');
		expect(request.headers.get('authorization')).toBe('Bearer test-key');
		expect(request.body).toMatchObject({
			model: 'gpt-5',
			store: false,
			prompt_cache_key: 'conv-1',
			max_output_tokens: 128_000,
		});
	});

	it('azure-openai-responses: the resource URL, the deployment, and store false', async () => {
		const request = await firstRequest('azure-openai-responses', 'gpt-5', {
			auth: { apiKey: 'azure-key' },
			env: {
				AZURE_OPENAI_RESOURCE_NAME: 'flue-test',
				AZURE_OPENAI_DEPLOYMENT_NAME_MAP: 'gpt-5=flue-gpt5, other=x',
			},
		});

		expect(request.url).toMatch(/^https:\/\/flue-test\.openai\.azure\.com\/openai\/v1\/responses/);
		expect(request.headers.get('api-key')).toBe('azure-key');
		expect(request.body).toMatchObject({ model: 'flue-gpt5', store: false });
	});

	it('openai-completions on the Cloudflare gateway: the URL from the account config and the gateway key only', async () => {
		const request = await firstRequest('cloudflare-ai-gateway', 'alibaba/qwen3-max', {
			auth: {
				headers: {
					'cf-aig-authorization': 'Bearer cf-key',
					Authorization: null,
					'x-api-key': null,
				},
			},
			env: { CLOUDFLARE_ACCOUNT_ID: 'acct', CLOUDFLARE_GATEWAY_ID: 'gw' },
		});

		expect(request.url).toBe(
			'https://gateway.ai.cloudflare.com/v1/acct/gw/compat/chat/completions',
		);
		expect(request.headers.get('cf-aig-authorization')).toBe('Bearer cf-key');
		expect(request.headers.get('authorization')).toBeNull();
		expect(request.body).toMatchObject({ model: 'alibaba/qwen3-max' });
	});

	it('google-generative-ai: the catalog URL with one version path', async () => {
		const request = await firstRequest('google', 'gemini-2.5-flash');

		expect(request.url).toBe(
			'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse',
		);
		expect(request.headers.get('x-goog-api-key')).toBe('test-key');
		expect(request.body).toMatchObject({
			generationConfig: { maxOutputTokens: 65_536 },
		});
	});

	it('google-vertex with an API key: the v1 express endpoint', async () => {
		const request = await firstRequest('google-vertex', 'gemini-2.5-flash');

		expect(request.url).toBe(
			'https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-flash:streamGenerateContent?alt=sse',
		);
		expect(request.headers.get('x-goog-api-key')).toBe('test-key');
	});

	it('mistral-conversations: the output cap leaves the context its safety margin', async () => {
		const request = await firstRequest('mistral', 'mistral-large-latest');

		expect(request.url).toBe('https://api.mistral.ai/v1/chat/completions');
		expect(request.headers.get('authorization')).toBe('Bearer test-key');
		// A 262,144-token window, minus the 4,096-token margin.
		expect(request.body).toMatchObject({ max_tokens: 258_048 });
	});

	it('bedrock-converse-stream: the bearer token and the output cap', async () => {
		const { sent, url, close } = await http2TestServer();
		try {
			await runTurn(
				flueAdapter('amazon-bedrock', 'amazon.nova-2-lite-v1:0', {
					auth: { baseUrl: url },
					env: { AWS_BEARER_TOKEN_BEDROCK: 'bedrock-token' },
				}),
			);
		} finally {
			await close();
		}

		const [request] = sent;
		expect(request?.url).toBe('/model/amazon.nova-2-lite-v1%3A0/converse-stream');
		expect(request?.headers.get('authorization')).toBe('Bearer bedrock-token');
		expect(request?.body).toMatchObject({
			inferenceConfig: { maxTokens: 65_535 },
		});
	});
});

describe('createModelAdapter session headers', () => {
	it('sends OpenCode its conversation header', async () => {
		const request = await firstRequest('opencode', 'claude-3-5-haiku');

		expect(request.url).toBe('https://opencode.ai/zen/v1/messages?beta=true');
		expect(request.headers.get('x-opencode-session')).toBe('conv-1');
	});

	it('sends the OpenRouter affinity header on the Anthropic wire', async () => {
		const request = await firstRequest('openrouter', 'anthropic/claude-sonnet-4.5');

		expect(request.headers.get('x-session-id')).toBe('conv-1');
	});
});

describe('createModelAdapter errors', () => {
	it('names the provider when it found no credential', () => {
		const { provider, model } = builtin('anthropic', 'claude-sonnet-4-5');

		expect(() =>
			createModelAdapter(provider, model, {
				auth: undefined,
				promptCacheKey: 'conv-1',
			}),
		).toThrow('No API key for provider: anthropic');
	});

	it('rejects an API that no adapter speaks', () => {
		const { provider, model } = builtin('anthropic', 'claude-sonnet-4-5');

		expect(() =>
			createModelAdapter(
				provider,
				{ ...model, api: 'custom-wire' },
				{ auth: keyAuth, promptCacheKey: 'conv-1' },
			),
		).toThrow('uses the "custom-wire" API');
	});
});

describe('clampMaxTokensToContext', () => {
	const model = { contextWindow: 200_000, maxTokens: 64_000 } satisfies Pick<
		FlueModel,
		'contextWindow' | 'maxTokens'
	>;

	it("keeps the model's cap when the context has room", () => {
		expect(clampMaxTokensToContext(model, 10_000)).toBe(64_000);
	});

	it('cuts the cap to what the context leaves free', () => {
		// 200,000 - 180,000 - 4,096
		expect(clampMaxTokensToContext(model, 180_000)).toBe(15_904);
	});

	it('keeps one token when the context is full', () => {
		expect(clampMaxTokensToContext(model, 300_000)).toBe(1);
	});

	it("keeps the model's cap when the context window is not known", () => {
		expect(clampMaxTokensToContext({ ...model, contextWindow: 0 }, 300_000)).toBe(64_000);
	});
});
