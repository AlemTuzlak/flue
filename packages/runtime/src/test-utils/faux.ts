/**
 * A fake model provider for tests, with pi's faux API. Queue assistant
 * messages (or functions that build them), and each model call answers with
 * the next one. TanStack's `fakeText` streams the answer, and the usage is
 * pi's estimate, so token-based behavior (overflow, caching) stays the same.
 *
 * ```ts
 * const faux = fauxProvider({ models: [{ id: 'model' }] });
 * faux.setResponses([fauxAssistantMessage([fauxText('Hello.')])]);
 * await start({ agents, providers: [faux.provider] });
 * ```
 */
import {
	EventType,
	type Modality,
	type ModelMessage,
	type RunErrorEvent,
	type TextOptions,
	type TokenUsage,
} from '@tanstack/ai';
import { type FakeResponse, FakeTextAdapter, fakeText } from '@tanstack/ai/testing';
import type {
	AgentMessage,
	AssistantMessage,
	ImageContent,
	JsonObject,
	StopReason,
	TextContent,
	ThinkingContent,
	ToolCall,
	ToolDeclaration,
} from '../llm-types.ts';
import { fromModelMessage } from '../model-messages.ts';
import { createProvider, type FlueModel } from '../providers/provider.ts';

const DEFAULT_PROVIDER = 'faux';
const DEFAULT_API = 'faux';
const DEFAULT_MODEL_ID = 'faux-1';
const EMPTY_QUEUE = 'No more faux responses queued';
const ZERO_USAGE = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

export interface FauxModelDefinition {
	id: string;
	name?: string;
	reasoning?: boolean;
	input?: ('text' | 'image')[];
	cost?: {
		input: number;
		output: number;
		cacheRead: number;
		cacheWrite: number;
	};
	contextWindow?: number;
	maxTokens?: number;
}

/** The request of one model call, in Flue's message shape. */
export interface FauxContext {
	systemPrompt?: string;
	messages: AgentMessage[];
	tools?: ToolDeclaration[];
}

export interface FauxRequestOptions {
	signal?: AbortSignal;
	/** The conversation's prompt cache key. */
	sessionId?: string;
}

export interface FauxProviderState {
	/** How many model calls the provider answered. */
	callCount: number;
}

export type FauxResponseFactory = (
	context: FauxContext,
	options: FauxRequestOptions,
	state: FauxProviderState,
	model: FlueModel,
) => AssistantMessage | Promise<AssistantMessage>;

export type FauxResponseStep = AssistantMessage | FauxResponseFactory;

export interface FauxProviderOptions {
	/** The provider id. The default is `faux`. */
	provider?: string;
	/** The model API of every model. The default is `faux`. */
	api?: string;
	models?: FauxModelDefinition[];
	/** Stream the answer at this many tokens (4 characters each) per second. */
	tokensPerSecond?: number;
}

export function fauxText(text: string): TextContent {
	return { type: 'text', text };
}

export function fauxThinking(thinking: string): ThinkingContent {
	return { type: 'thinking', thinking };
}

export function fauxToolCall(
	name: string,
	args: JsonObject,
	options: { id?: string } = {},
): ToolCall {
	return {
		type: 'toolCall',
		id: options.id ?? randomId('tool'),
		name,
		arguments: args,
	};
}

type FauxBlock = TextContent | ThinkingContent | ToolCall;

/** A scripted answer. `content` is a string, one block, or a list of blocks. */
export function fauxAssistantMessage(
	content: string | FauxBlock | FauxBlock[],
	options: {
		stopReason?: StopReason;
		errorMessage?: string;
		responseId?: string;
		timestamp?: number;
	} = {},
): AssistantMessage {
	const blocks =
		typeof content === 'string'
			? [fauxText(content)]
			: Array.isArray(content)
				? content
				: [content];
	return {
		role: 'assistant',
		content: blocks,
		api: DEFAULT_API,
		provider: DEFAULT_PROVIDER,
		model: DEFAULT_MODEL_ID,
		usage: ZERO_USAGE,
		stopReason: options.stopReason ?? 'stop',
		...(options.errorMessage === undefined ? {} : { errorMessage: options.errorMessage }),
		...(options.responseId === undefined ? {} : { responseId: options.responseId }),
		timestamp: options.timestamp ?? Date.now(),
	};
}

/** A faux provider and the controls of its response queue. */
export function fauxProvider(options: FauxProviderOptions = {}) {
	const providerId = options.provider ?? DEFAULT_PROVIDER;
	const api = options.api ?? DEFAULT_API;
	const definitions = options.models?.length
		? options.models
		: [{ id: DEFAULT_MODEL_ID, name: 'Faux Model' }];
	const models: FlueModel[] = definitions.map((definition) => ({
		id: definition.id,
		name: definition.name ?? definition.id,
		api,
		provider: providerId,
		baseUrl: 'http://localhost:0',
		reasoning: definition.reasoning ?? false,
		input: definition.input ?? ['text', 'image'],
		cost: definition.cost ?? {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
		},
		contextWindow: definition.contextWindow ?? 128_000,
		maxTokens: definition.maxTokens ?? 16_384,
	}));
	const queue: FauxResponseStep[] = [];
	const core: FauxCore = {
		queue,
		state: { callCount: 0 },
		prompts: new Map(),
		tokensPerSecond: options.tokensPerSecond,
	};
	const provider = createProvider({
		id: providerId,
		name: 'Faux',
		auth: { apiKey: { name: 'Faux', resolve: async () => ({ auth: {} }) } },
		models,
		createAdapter: (model) => new FauxTextAdapter(model, core),
	});
	return {
		provider,
		api,
		models,
		state: core.state,
		getModel: (modelId?: string) =>
			modelId === undefined ? models[0] : models.find((model) => model.id === modelId),
		setResponses(responses: FauxResponseStep[]) {
			queue.splice(0, queue.length, ...responses);
		},
		appendResponses(responses: FauxResponseStep[]) {
			queue.push(...responses);
		},
		getPendingResponseCount: () => queue.length,
	};
}

interface FauxCore {
	queue: FauxResponseStep[];
	state: FauxProviderState;
	/** The last prompt of each cache key, for pi's cache estimate. */
	prompts: Map<string, string>;
	tokensPerSecond: number | undefined;
}

class FauxTextAdapter extends FakeTextAdapter<string, readonly Modality[]> {
	constructor(
		private readonly flueModel: FlueModel,
		private readonly core: FauxCore,
	) {
		super(flueModel.id, {});
	}

	override async *chatStream(options: TextOptions) {
		const step = this.core.queue.shift();
		this.core.state.callCount += 1;
		const context = fauxContext(options);
		const message = await this.answer(step, context, options);
		const usage = estimateUsage(context, message, options, this.core.prompts);
		const response = message ? toFakeResponse(message) : { error: EMPTY_QUEUE };
		const inner = fakeText({
			model: this.flueModel.id,
			...(this.core.tokensPerSecond ? { tokensPerSecond: this.core.tokensPerSecond } : {}),
		});
		inner.setResponses([response]);
		for await (const chunk of inner.chatStream(options)) {
			if (chunk.type === EventType.RUN_ERROR) {
				yield { ...chunk, usage };
			} else if (chunk.type === EventType.RUN_FINISHED && message?.stopReason === 'aborted') {
				const errorMessage = message.errorMessage ?? 'Request was aborted';
				yield {
					type: EventType.RUN_ERROR,
					message: errorMessage,
					code: 'aborted',
					error: { message: errorMessage, code: 'aborted' },
					usage,
					timestamp: Date.now(),
				} satisfies RunErrorEvent;
			} else if (chunk.type === EventType.RUN_FINISHED) {
				yield {
					...chunk,
					usage,
					...(message?.responseId ? { responseId: message.responseId } : {}),
				};
			} else {
				yield chunk;
			}
		}
	}

	private async answer(
		step: FauxResponseStep | undefined,
		context: FauxContext,
		options: TextOptions,
	) {
		if (typeof step !== 'function') return step;
		const requestOptions: FauxRequestOptions = {
			...(options.abortController ? { signal: options.abortController.signal } : {}),
			...(options.promptCache?.key ? { sessionId: options.promptCache.key } : {}),
		};
		try {
			return await step(context, requestOptions, this.core.state, this.flueModel);
		} catch (error) {
			return fauxAssistantMessage([], {
				stopReason: 'error',
				errorMessage: error instanceof Error ? error.message : String(error),
			});
		}
	}
}

function fauxContext(options: TextOptions): FauxContext {
	const systemPrompt = (options.systemPrompts ?? [])
		.map((prompt) => (typeof prompt === 'string' ? prompt : prompt.content))
		.join('\n\n');
	return {
		...(systemPrompt ? { systemPrompt } : {}),
		messages: options.messages.map((message: ModelMessage) => fromModelMessage(message)),
		...(options.tools?.length
			? {
					tools: options.tools.map((tool) => ({
						name: tool.name,
						description: tool.description,
						parameters: tool.inputSchema ?? {},
					})),
				}
			: {}),
	};
}

/** pi's step shape as a `fakeText` answer. Blocks of one kind join, in pi's default order. */
function toFakeResponse(message: AssistantMessage): FakeResponse {
	switch (message.stopReason) {
		case 'error':
			return { error: message.errorMessage ?? 'Faux response failed' };
		case 'pending':
			return { error: 'Faux response ended without a stop reason' };
		case 'deferred':
			return { error: 'Faux deferred responses are not supported' };
	}
	const text = message.content
		.flatMap((block) => (block.type === 'text' ? [block.text] : []))
		.join('');
	const thinking = message.content
		.flatMap((block) => (block.type === 'thinking' ? [block.thinking] : []))
		.join('');
	const toolCalls = message.content.flatMap((block) =>
		block.type === 'toolCall' ? [{ id: block.id, name: block.name, input: block.arguments }] : [],
	);
	const finishReason = {
		stop: 'stop',
		aborted: 'stop',
		length: 'length',
		toolUse: 'tool_calls',
	} as const;
	return {
		...(text ? { text } : {}),
		...(thinking ? { thinking } : {}),
		...(toolCalls.length > 0 ? { toolCalls } : {}),
		finishReason: finishReason[message.stopReason],
	};
}

// ─── pi's usage estimate ────────────────────────────────────────────────────

function estimateTokens(text: string) {
	return Math.ceil(text.length / 4);
}

function contentToText(content: string | readonly (TextContent | ImageContent)[]) {
	if (typeof content === 'string') return content;
	return content
		.map((block) =>
			block.type === 'text' ? block.text : `[image:${block.mimeType}:${block.data.length}]`,
		)
		.join('\n');
}

function assistantContentToText(content: AssistantMessage['content']) {
	return content
		.map((block) => {
			if (block.type === 'text') return block.text;
			if (block.type === 'thinking') return block.thinking;
			return `${block.name}:${JSON.stringify(block.arguments)}`;
		})
		.join('\n');
}

function messageToText(message: AgentMessage) {
	switch (message.role) {
		case 'user':
			return contentToText(message.content);
		case 'assistant':
			return assistantContentToText(message.content);
		case 'toolResult':
			return [message.toolName, ...message.content.map((block) => contentToText([block]))].join(
				'\n',
			);
		case 'system':
			return typeof message.content === 'string' ? message.content : contentToText(message.content);
		case 'signal':
			return message.content;
	}
}

/** pi's prompt text: the system prompt as its leading system message, then each message, a blank line apart. */
function serializeContext(context: FauxContext) {
	const system = context.systemPrompt === undefined ? [] : [`system:${context.systemPrompt}`];
	return [
		...system,
		...context.messages.map((message) => `${message.role}:${messageToText(message)}`),
	].join('\n\n');
}

function commonPrefixLength(a: string, b: string) {
	const length = Math.min(a.length, b.length);
	let index = 0;
	while (index < length && a[index] === b[index]) index++;
	return index;
}

/**
 * pi's faux usage, in TanStack's shape (`promptTokens` holds the cached
 * parts). With a cache key, the part of the prompt that matches the key's
 * previous prompt reads as cached.
 */
function estimateUsage(
	context: FauxContext,
	message: AssistantMessage | undefined,
	options: TextOptions,
	prompts: Map<string, string>,
): TokenUsage {
	const promptText = serializeContext(context);
	const promptTokens = estimateTokens(promptText);
	const output = message ? estimateTokens(assistantContentToText(message.content)) : 0;
	let input = promptTokens;
	let cacheRead = 0;
	let cacheWrite = 0;
	const key = options.promptCache?.key;
	if (key && options.promptCache?.retention !== 'none') {
		const previous = prompts.get(key);
		if (previous) {
			const cachedChars = commonPrefixLength(previous, promptText);
			cacheRead = estimateTokens(previous.slice(0, cachedChars));
			cacheWrite = estimateTokens(promptText.slice(cachedChars));
			input = Math.max(0, promptTokens - cacheRead);
		} else {
			cacheWrite = promptTokens;
		}
		prompts.set(key, promptText);
	}
	return {
		promptTokens: input + cacheRead + cacheWrite,
		completionTokens: output,
		totalTokens: input + output + cacheRead + cacheWrite,
		...(cacheRead || cacheWrite
			? {
					promptTokensDetails: {
						cachedTokens: cacheRead,
						cacheWriteTokens: cacheWrite,
					},
				}
			: {}),
	};
}

function randomId(prefix: string) {
	return `${prefix}:${Date.now()}:${Math.random().toString(36).slice(2)}`;
}
