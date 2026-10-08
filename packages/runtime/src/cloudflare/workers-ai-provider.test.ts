import { chat, EventType, type ModelMessage, toolDefinition } from '@tanstack/ai';
import { describe, expect, it } from 'vitest';
import type { ThinkingLevel } from '../llm-types.ts';
import { AssistantStreamAssembler, modelInfo } from '../model-messages.ts';
import type { FlueModel } from '../providers/provider.ts';
import {
	type CloudflareBindingProviderOptions,
	cloudflareBindingProvider,
} from './workers-ai-provider.ts';

function sseResponse(chunks: unknown[]): Response {
	const body = `${chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join('')}data: [DONE]\n\n`;
	return new Response(body, {
		headers: { 'content-type': 'text/event-stream' },
	});
}

/** A minimal valid Anthropic Messages SSE stream, with `event:` names. */
function anthropicSseResponse(): Response {
	const events = [
		[
			'message_start',
			{
				type: 'message_start',
				message: { id: 'msg_1', usage: { input_tokens: 1, output_tokens: 0 } },
			},
		],
		[
			'content_block_start',
			{
				type: 'content_block_start',
				index: 0,
				content_block: { type: 'text', text: '' },
			},
		],
		[
			'content_block_delta',
			{
				type: 'content_block_delta',
				index: 0,
				delta: { type: 'text_delta', text: 'hi' },
			},
		],
		['content_block_stop', { type: 'content_block_stop', index: 0 }],
		[
			'message_delta',
			{
				type: 'message_delta',
				delta: { stop_reason: 'end_turn', stop_sequence: null },
				usage: { output_tokens: 1 },
			},
		],
		['message_stop', { type: 'message_stop' }],
	] as const;
	const body = events
		.map(([event, data]) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
		.join('');
	return new Response(body, {
		headers: { 'content-type': 'text/event-stream' },
	});
}

/** Minimal valid OpenAI Responses SSE (message with one output_text part). */
function responsesSseResponse(text = 'hi'): Response {
	const events = [
		{ type: 'response.created', response: { id: 'resp_1' } },
		{
			type: 'response.output_item.added',
			output_index: 0,
			item: {
				id: 'msg_1',
				type: 'message',
				role: 'assistant',
				status: 'in_progress',
				content: [],
			},
		},
		{
			type: 'response.content_part.added',
			item_id: 'msg_1',
			output_index: 0,
			content_index: 0,
			part: { type: 'output_text', text: '' },
		},
		{
			type: 'response.output_text.delta',
			item_id: 'msg_1',
			output_index: 0,
			content_index: 0,
			delta: text,
		},
		{
			type: 'response.output_text.done',
			item_id: 'msg_1',
			output_index: 0,
			content_index: 0,
			text,
		},
		{
			type: 'response.content_part.done',
			item_id: 'msg_1',
			output_index: 0,
			content_index: 0,
			part: { type: 'output_text', text },
		},
		{
			type: 'response.output_item.done',
			output_index: 0,
			item: {
				id: 'msg_1',
				type: 'message',
				role: 'assistant',
				status: 'completed',
				content: [{ type: 'output_text', text }],
			},
		},
		{
			type: 'response.completed',
			response: { id: 'resp_1', status: 'completed', output: [] },
		},
	] as const;
	const body = events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join('');
	return new Response(body, {
		headers: { 'content-type': 'text/event-stream' },
	});
}

interface BindingRun {
	modelId: string;
	params: Record<string, unknown>;
	extraHeaders: Record<string, string> | undefined;
}

/** A binding that records each `run` and answers with `respond()`. */
function recordingProvider(
	respond: () => Response,
	options: Partial<CloudflareBindingProviderOptions> = {},
) {
	const runs: BindingRun[] = [];
	const binding = {
		async run(
			modelId: string,
			params: Record<string, unknown>,
			runOptions?: Record<string, unknown>,
		) {
			const extraHeaders = runOptions?.extraHeaders;
			runs.push({
				modelId,
				params,
				extraHeaders:
					typeof extraHeaders === 'object' && extraHeaders !== null
						? Object.fromEntries(
								Object.entries(extraHeaders).map(([name, value]) => [name, String(value)]),
							)
						: undefined,
			});
			return respond();
		},
	};
	const provider = cloudflareBindingProvider({
		binding,
		gateway: false,
		...options,
	});
	return { provider, runs, last: () => runs.at(-1) };
}

function catalogModel(
	provider: ReturnType<typeof cloudflareBindingProvider>,
	matches: (model: FlueModel) => boolean,
) {
	const model = provider.getModels().find(matches);
	if (!model) throw new Error('Expected a catalog model.');
	return model;
}

function bindingModel(overrides: Partial<FlueModel>): FlueModel {
	return {
		id: '@cf/test/model',
		name: 'Test Model',
		api: 'openai-completions',
		provider: 'cloudflare',
		baseUrl: '',
		reasoning: false,
		input: ['text'],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 0,
		maxTokens: 0,
		...overrides,
	};
}

/** One model call through the provider's adapter; the answer as Flue's assistant message. */
async function runOnce(
	provider: ReturnType<typeof cloudflareBindingProvider>,
	model: FlueModel,
	request: {
		systemPrompt?: string;
		messages?: ModelMessage[];
		tools?: string[];
		reasoning?: ThinkingLevel;
	} = {},
) {
	const { createAdapter } = provider;
	if (!createAdapter) throw new Error('The binding provider has no adapter.');
	const adapter = createAdapter(model, {
		auth: { auth: {} },
		promptCacheKey: 'session-1',
	});
	const assembler = new AssistantStreamAssembler(modelInfo(model));
	try {
		for await (const chunk of chat({
			adapter,
			messages: request.messages ?? [{ role: 'user', content: 'hi' }],
			systemPrompts: request.systemPrompt ? [request.systemPrompt] : [],
			tools: (request.tools ?? []).map((name) =>
				toolDefinition({
					name,
					description: `Does ${name} things.`,
					inputSchema: { type: 'object', properties: {} },
				}),
			),
			...(request.reasoning ? { reasoning: { level: request.reasoning, summary: true } } : {}),
			// One model call: tool calls stay unanswered.
			agentLoopStrategy: () => false,
		})) {
			assembler.push(chunk);
		}
	} catch (error) {
		assembler.push({
			type: EventType.RUN_ERROR,
			timestamp: Date.now(),
			message: error instanceof Error ? error.message : String(error),
		});
	}
	return assembler.finish();
}

describe('Cloudflare Workers AI assistant content', () => {
	it.each([
		['null', { role: 'assistant', content: null }],
		['omitted', { role: 'assistant' }],
	])('treats %s content as no text and continues processing tool calls', async (_name, delta) => {
		const { provider } = recordingProvider(() =>
			sseResponse([
				{ choices: [{ delta }] },
				{
					choices: [
						{
							delta: {
								tool_calls: [
									{
										index: 0,
										id: 'call_1',
										function: { name: 'lookup', arguments: '{}' },
									},
								],
							},
						},
					],
				},
				{ choices: [{ delta: {}, finish_reason: 'tool_calls' }] },
			]),
		);
		const model = catalogModel(provider, (entry) => entry.id.startsWith('@cf/'));

		const result = await runOnce(provider, model, { tools: ['lookup'] });

		expect(result.stopReason).toBe('toolUse');
		expect(result.content).toMatchObject([
			{ type: 'toolCall', id: 'call_1', name: 'lookup', arguments: {} },
		]);
	});

	it.each([
		['an object', { unexpected: true }],
		['an array', ['unexpected']],
	])('rejects content containing %s', async (description, content) => {
		const { provider } = recordingProvider(() =>
			sseResponse([
				{ choices: [{ delta: { role: 'assistant', content } }] },
				{ choices: [{ delta: {}, finish_reason: 'stop' }] },
			]),
		);
		const model = catalogModel(provider, (entry) => entry.id.startsWith('@cf/'));

		const result = await runOnce(provider, model);

		expect(result.stopReason).toBe('error');
		expect(result.errorMessage).toContain(
			`invalid choices[0].delta.content: expected a string, null, or an omitted field; received ${description}`,
		);
	});
});

describe('Cloudflare binding Anthropic gateway effort', () => {
	function anthropic(options?: Partial<CloudflareBindingProviderOptions>) {
		const recording = recordingProvider(anthropicSseResponse, options);
		const model = catalogModel(
			recording.provider,
			(entry) => entry.id === 'anthropic/claude-opus-5',
		);
		return { ...recording, model };
	}

	it('maps the thinking level to output_config.effort for adaptive-thinking models', async () => {
		const { provider, model, last } = anthropic();
		const result = await runOnce(provider, model, {
			systemPrompt: 'x',
			reasoning: 'low',
		});

		expect(result.errorMessage).toBeUndefined();
		expect(result.content).toMatchObject([{ type: 'text', text: 'hi' }]);
		expect(last()?.modelId).toBe('anthropic/claude-opus-5');
		expect(last()?.params).toMatchObject({
			thinking: { type: 'adaptive', display: 'summarized' },
			output_config: { effort: 'low' },
		});
	});

	it('maps xhigh through the model reasoning map', async () => {
		const { provider, model, last } = anthropic();
		const result = await runOnce(provider, model, {
			systemPrompt: 'x',
			reasoning: 'xhigh',
		});

		expect(result.errorMessage).toBeUndefined();
		expect(last()?.params).toMatchObject({ output_config: { effort: 'xhigh' } });
	});

	it('omits output_config when reasoning is not set', async () => {
		const { provider, model, last } = anthropic();
		const result = await runOnce(provider, model, { systemPrompt: 'x' });

		expect(result.errorMessage).toBeUndefined();
		expect(last()?.params.output_config).toBeUndefined();
	});
});

describe('Cloudflare binding Anthropic prompt caching', () => {
	it('defaults to cacheRetention none — no cache_control markers', async () => {
		const { provider, last } = recordingProvider(anthropicSseResponse);
		const model = catalogModel(provider, (entry) => entry.id === 'anthropic/claude-opus-5');
		await runOnce(provider, model, { systemPrompt: 'x' });
		expect(JSON.stringify(last()?.params)).not.toContain('cache_control');
	});

	it('cacheRetention short emits cache_control', async () => {
		const { provider, last } = recordingProvider(anthropicSseResponse, {
			cacheRetention: 'short',
		});
		const model = catalogModel(provider, (entry) => entry.id === 'anthropic/claude-opus-5');
		await runOnce(provider, model, { systemPrompt: 'x' });
		const payload = JSON.stringify(last()?.params);
		expect(payload).toContain('cache_control');
		expect(payload).toContain('"type":"ephemeral"');
	});
});

describe('Cloudflare binding Anthropic request headers', () => {
	it('sends betas as the anthropic-beta header, not in the request body, with the session affinity', async () => {
		const { provider, last } = recordingProvider(anthropicSseResponse);
		// Budget thinking with tools turns on the interleaved-thinking beta.
		const model = catalogModel(provider, (entry) => entry.id === 'anthropic/claude-sonnet-4-5');
		const result = await runOnce(provider, model, {
			systemPrompt: 'x',
			tools: ['lookup'],
			reasoning: 'medium',
		});

		expect(result.errorMessage).toBeUndefined();
		expect(last()?.params).not.toHaveProperty('betas');
		expect(last()?.params.tools).toHaveLength(1);
		expect(last()?.extraHeaders).toMatchObject({
			'anthropic-beta': expect.stringContaining('interleaved-thinking'),
			'x-session-affinity': 'session-1',
		});
	});

	it('sends no anthropic-beta header when no beta features apply', async () => {
		const { provider, last } = recordingProvider(anthropicSseResponse);
		const model = catalogModel(provider, (entry) => entry.id === 'anthropic/claude-opus-5');
		await runOnce(provider, model, { systemPrompt: 'x' });

		expect(last()?.params).not.toHaveProperty('betas');
		expect(last()?.extraHeaders?.['anthropic-beta']).toBeUndefined();
	});
});

describe('Cloudflare binding transcript adaptation', () => {
	it('carries request-level tools and the system prompt on the chat-completions branch', async () => {
		const { provider, last } = recordingProvider(() =>
			sseResponse([
				{ choices: [{ delta: { content: 'ok' } }] },
				{ choices: [{ delta: {}, finish_reason: 'stop' }] },
			]),
		);
		const result = await runOnce(provider, bindingModel({}), {
			systemPrompt: 'System instructions.',
			tools: ['lookup'],
		});

		expect(result.errorMessage).toBeUndefined();
		const tools = last()?.params.tools as Array<{
			function?: { name?: string; description?: string };
		}>;
		expect(tools.map((tool) => tool.function?.name)).toContain('lookup');
		expect(tools[0]?.function?.description).toContain('Does lookup things.');
		const messages = last()?.params.messages as Array<{ role?: string }>;
		expect(messages[0]?.role).toBe('system');
		expect(JSON.stringify(messages[0])).toContain('System instructions.');
		expect(last()?.extraHeaders).toMatchObject({
			'x-session-affinity': 'session-1',
		});
	});

	it('merges per-model catalog compat overrides over the base Workers AI profile', async () => {
		const { provider, last } = recordingProvider(() =>
			sseResponse([
				{ choices: [{ delta: { content: 'ok' } }] },
				{ choices: [{ delta: {}, finish_reason: 'stop' }] },
			]),
		);
		// `supportsDeveloperRole` is false in the base profile; a catalog
		// override must win and switch the instruction role.
		await runOnce(
			provider,
			bindingModel({
				reasoning: true,
				compat: { supportsDeveloperRole: true, supportsReasoningEffort: true },
			}),
			{ systemPrompt: 'System instructions.' },
		);
		const messages = last()?.params.messages as Array<{ role?: string }>;
		expect(messages[0]?.role).toBe('developer');
	});

	it('carries request-level tools and the system prompt on the Responses branch', async () => {
		const { provider, last } = recordingProvider(() => responsesSseResponse());
		const result = await runOnce(
			provider,
			bindingModel({ id: 'openai/test-model', api: 'openai-responses' }),
			{ systemPrompt: 'System instructions.', tools: ['lookup'] },
		);

		expect(result.errorMessage).toBeUndefined();
		const tools = last()?.params.tools as Array<{ name?: string }> | undefined;
		expect(tools?.map((tool) => tool.name)).toContain('lookup');
		// TanStack sends the system prompt as `instructions` (pi put it in `input`).
		expect(JSON.stringify(last()?.params)).toContain('System instructions.');
	});

	it('renders mid-conversation tool additions as additional_tools on the Responses branch', async () => {
		const { provider, last } = recordingProvider(() => responsesSseResponse());
		// Tool `a` was declared by the earlier call; tool `b` is new. The
		// deferred-loading channel (`additional_tools`) is the model's own, so
		// `b` must not be in the request-level tools.
		await runOnce(
			provider,
			bindingModel({
				id: 'openai/gpt-5.4',
				api: 'openai-responses',
				compat: { supportsAdditionalTools: true },
			}),
			{
				systemPrompt: 'System instructions.',
				tools: ['a', 'b'],
				messages: [
					{ role: 'user', content: 'hi' },
					{
						role: 'assistant',
						content: 'ok',
						midConversationChange: {
							tools: ['a'],
							systemPrompts: [],
						},
					},
					{ role: 'user', content: 'more' },
				],
			},
		);
		const tools = (last()?.params.tools ?? []) as Array<{ name?: string }>;
		expect(tools.map((tool) => tool.name)).toEqual(['a']);
		expect(JSON.stringify(last()?.params.input)).toContain('"type":"additional_tools"');
		expect(JSON.stringify(last()?.params.input)).toContain('"b"');
	});

	it('maps the thinking level into the Responses reasoning field', async () => {
		const { provider, last } = recordingProvider(() => responsesSseResponse());
		await runOnce(
			provider,
			bindingModel({
				id: 'openai/test-model',
				api: 'openai-responses',
				reasoning: true,
				reasoningMap: { low: 'low', medium: 'medium', high: 'high' },
			}),
			{ reasoning: 'high' },
		);
		expect(last()?.params.reasoning).toEqual({ effort: 'high', summary: 'auto' });
	});
});

describe('Cloudflare binding diagnostics', () => {
	it("keeps the response's gateway log id and the provider finish value", async () => {
		const { provider } = recordingProvider(
			() =>
				new Response(
					`data: ${JSON.stringify({ choices: [{ delta: { content: 'ok' } }] })}\n\ndata: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: 'stop' }] })}\n\ndata: [DONE]\n\n`,
					{
						headers: {
							'content-type': 'text/event-stream',
							'cf-aig-log-id': 'log-123',
						},
					},
				),
		);
		const result = await runOnce(provider, bindingModel({}));

		expect(result.diagnostics).toMatchObject([
			{
				type: 'flue:provider_response',
				details: { gatewayLogId: 'log-123', providerFinishReason: 'stop' },
			},
		]);
	});
});
