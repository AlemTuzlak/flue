/**
 * The model boundary. Flue messages become TanStack `ModelMessage`s for a
 * request, and the TanStack stream chunks of a response become Flue assistant
 * blocks, with the same block events that pi streamed.
 *
 * TanStack's chat engine does the cross-model replay (foreign signatures,
 * tool call ids, failed turns) from the source tag and the stop reason that
 * this module puts on each assistant message.
 */
import {
	type ContentPart,
	EventType,
	fromSpecTokenUsage,
	type ModelMessage,
	type ToolCall as ModelToolCall,
	type RunErrorEvent,
	type RunFinishedEvent,
	type StreamChunk,
	type TokenUsage,
} from '@tanstack/ai';
import {
	type BlockOrderEntry,
	buildBlockOrder,
	type OrderedAssistantBlock,
	orderedAssistantBlocks,
	REDACTED_THINKING_ID_PREFIX,
	tanstackMetadata,
} from '@tanstack/ai/adapter-internals';
import { type ModelCostRates, modelCost } from '@tanstack/ai-models';
import {
	type DocumentContextBlock,
	documentOmittedPlaceholder,
	isDocumentContextBlock,
	NATIVE_DOCUMENT_APIS,
	warnDocumentsOmitted,
} from './document-attachments.ts';
import type {
	AgentMessage,
	AssistantMessage,
	ImageContent,
	JsonObject,
	TextContent,
	ThinkingContent,
	ToolCall,
	ToolDeclaration,
	ToolResultMessage,
	Usage,
	UserMessage,
} from './llm-types.ts';
import { renderSignalMessage, toolResultText } from './message-rendering.ts';
import {
	attachProviderResponseDiagnostics,
	chunkProviderResponseDiagnostics,
} from './provider-diagnostics.ts';
import type { FlueModel } from './providers/provider.ts';

/** What the boundary needs to know about the model of a request. */
export interface FlueModelInfo {
	provider: string;
	api: string;
	id: string;
	input: readonly ('text' | 'image' | 'document')[];
	/** Prices per 1M tokens. Without them, usage costs are 0. */
	cost?: ModelCostRates;
}

const USER_IMAGE_PLACEHOLDER = '(image omitted: model does not support images)';
const TOOL_IMAGE_PLACEHOLDER = '(tool image omitted: model does not support images)';
const REDACTED_THINKING_TEXT = '[Reasoning redacted]';

/** What Flue's model message mapping needs to know about `model`. */
export function modelInfo(model: FlueModel): FlueModelInfo {
	const input = model.input.filter(
		(kind): kind is 'text' | 'image' => kind === 'text' || kind === 'image',
	);
	return {
		provider: model.provider,
		api: model.api,
		id: model.id,
		// Documents follow the API, not the catalog: see `NATIVE_DOCUMENT_APIS`.
		input: NATIVE_DOCUMENT_APIS.has(model.api) ? [...input, 'document'] : input,
		cost: model.cost,
	};
}

/**
 * Build the TanStack request for one model call.
 *
 * `messages` is the conversation in Flue's shape. Signals render as user
 * text, and media that `target` cannot read becomes a text placeholder.
 */
export function toModelRequest(
	input: {
		systemPrompt: string;
		messages: readonly AgentMessage[];
		tools: readonly ToolDeclaration[];
	},
	target: FlueModelInfo,
) {
	return {
		systemPrompts: input.systemPrompt ? [input.systemPrompt] : [],
		messages: input.messages.flatMap((message) => toModelMessages(message, target)),
		tools: input.tools.map(({ name, description, parameters }) => ({
			name,
			description,
			parameters,
		})),
	};
}

function toModelMessages(message: AgentMessage, target: FlueModelInfo): ModelMessage[] {
	switch (message.role) {
		// ponytail: a `system` message is pi's transcript form of the prompt; the prompt comes in `systemPrompt`.
		case 'system':
			return [];
		case 'signal':
			return [{ role: 'user', content: renderSignalMessage(message) }];
		case 'user':
			return [
				{
					role: 'user',
					content:
						typeof message.content === 'string'
							? message.content
							: toContentParts(message.content, target, USER_IMAGE_PLACEHOLDER),
				},
			];
		case 'assistant':
			return [toAssistantModelMessage(message)];
		case 'toolResult': {
			const content = toContentParts(message.content, target, TOOL_IMAGE_PLACEHOLDER);
			const onlyText =
				content.length === 1 && content[0]?.type === 'text' ? content[0].content : undefined;
			return [
				{
					role: 'tool',
					toolCallId: message.toolCallId,
					name: message.toolName,
					content: onlyText ?? content,
					...(message.isError ? { error: toolResultText(message.content) } : {}),
				},
			];
		}
	}
}

function toContentParts(
	blocks: readonly (TextContent | ImageContent)[],
	target: FlueModelInfo,
	imagePlaceholder: string,
) {
	return readableParts(blocks.map(toContentPart), target, imagePlaceholder);
}

/** The parts that `target` can read: other media becomes a text placeholder. */
function readableParts(
	parts: readonly ContentPart[],
	target: FlueModelInfo,
	imagePlaceholder: string,
) {
	const readable: ContentPart[] = [];
	for (const source of parts) {
		const part = readablePart(source, target, imagePlaceholder);
		const last = readable.at(-1);
		// pi folds a run of omitted images into one placeholder.
		const isRepeatedPlaceholder =
			part.type === 'text' &&
			part.content === imagePlaceholder &&
			last?.type === 'text' &&
			last.content === imagePlaceholder;
		if (!isRepeatedPlaceholder) readable.push(part);
	}
	return readable;
}

function readablePart(
	part: ContentPart,
	target: FlueModelInfo,
	imagePlaceholder: string,
): ContentPart {
	if (part.type === 'document' && !target.input.includes('document')) {
		warnDocumentsOmitted(target.api);
		const filename = isJsonObject(part.metadata) ? part.metadata.filename : undefined;
		return {
			type: 'text',
			content: documentOmittedPlaceholder(
				target.api,
				typeof filename === 'string' ? filename : undefined,
			),
		};
	}
	if (part.type === 'image' && !target.input.includes('image'))
		return { type: 'text', content: imagePlaceholder };
	return part;
}

function toContentPart(block: TextContent | ImageContent): ContentPart {
	if (block.type === 'text') return { type: 'text', content: block.text };
	const source = {
		type: 'data' as const,
		value: block.data,
		mimeType: block.mimeType,
	};
	if (isDocumentContextBlock(block))
		return {
			type: 'document',
			source,
			...(block.filename ? { metadata: { filename: block.filename } } : {}),
		};
	return { type: 'image', source };
}

/**
 * A user or signal message as a TanStack harness input: the text, and the
 * image and document parts as they are. Each model call converts the media
 * that its model cannot read (see {@link toModelContext}).
 */
export function toUserInput(message: AgentMessage) {
	switch (message.role) {
		case 'signal':
			return renderSignalMessage(message);
		case 'user': {
			if (typeof message.content === 'string') return message.content;
			const [only] = message.content;
			// One text block is plain text, as the transcript keeps a typed message.
			if (message.content.length === 1 && only?.type === 'text') return only.text;
			return message.content.map(toContentPart);
		}
		default:
			throw new Error(`[flue] A ${message.role} message cannot be a harness input.`);
	}
}

/**
 * The messages of one model call for `target`, with the media rules of
 * {@link toModelRequest}: image and document parts of user and tool messages
 * that `target` cannot read become text placeholders. Other fields of each
 * message stay, so the mid-conversation records of assistant messages stay.
 */
export function toModelContext(messages: readonly ModelMessage[], target: FlueModelInfo) {
	return messages.map((message): ModelMessage => {
		if (message.role === 'assistant' || !Array.isArray(message.content)) return message;
		const placeholder = message.role === 'user' ? USER_IMAGE_PLACEHOLDER : TOOL_IMAGE_PLACEHOLDER;
		return { ...message, content: readableParts(message.content, target, placeholder) };
	});
}

function toAssistantModelMessage(message: AssistantMessage): ModelMessage {
	const thinking: NonNullable<ModelMessage['thinking']> = [];
	const toolCalls: ModelToolCall[] = [];
	const order: BlockOrderEntry[] = [];
	let text = '';
	for (const block of message.content) {
		switch (block.type) {
			case 'text':
				text += block.text;
				order.push({ type: 'text', text: block.text });
				break;
			case 'thinking':
				// TanStack sends no thinking block that has neither text nor a signature.
				if (!block.thinking && !block.thinkingSignature) break;
				order.push({ type: 'thinking' });
				thinking.push({
					content: block.redacted ? '' : block.thinking,
					...(block.thinkingSignature ? { signature: block.thinkingSignature } : {}),
					...(block.redacted ? { redacted: true } : {}),
				});
				break;
			case 'toolCall':
				order.push({ type: 'tool-call', id: block.id });
				toolCalls.push({
					id: block.id,
					type: 'function',
					function: {
						name: block.name,
						arguments: JSON.stringify(block.arguments),
					},
					...(block.thoughtSignature
						? { metadata: { thoughtSignature: block.thoughtSignature } }
						: {}),
				});
				break;
		}
	}
	const blockOrder = buildBlockOrder(order);
	const isFailed = message.stopReason === 'error' || message.stopReason === 'aborted';
	return {
		role: 'assistant',
		content: text || null,
		...(thinking.length > 0 ? { thinking } : {}),
		...(toolCalls.length > 0 ? { toolCalls } : {}),
		...(blockOrder ? { blockOrder } : {}),
		metadata: {
			tanstack: {
				source: {
					provider: message.provider,
					api: message.api,
					model: message.model,
				},
				...(isFailed ? { stopReason: message.stopReason } : {}),
			},
		},
	};
}

/**
 * A TanStack `ModelMessage` in Flue's shape: the reverse of
 * {@link toModelRequest}. A message holds no usage, so usage reads as zero.
 */
export function fromModelMessage(message: ModelMessage) {
	const timestamp = message.createdAt?.getTime() ?? Date.now();
	switch (message.role) {
		case 'user':
			return {
				role: 'user',
				content:
					typeof message.content === 'string'
						? message.content
						: fromContentParts(message.content ?? []),
				timestamp,
			} satisfies UserMessage;
		case 'tool':
			return {
				role: 'toolResult',
				toolCallId: message.toolCallId ?? '',
				toolName: message.name ?? '',
				content:
					typeof message.content === 'string'
						? [{ type: 'text', text: message.content }]
						: fromContentParts(message.content ?? []),
				isError: message.error !== undefined,
				timestamp,
			} satisfies ToolResultMessage;
		case 'assistant':
			return fromAssistantModelMessage(message, timestamp);
	}
}

function fromContentParts(parts: readonly ContentPart[]) {
	const blocks: (TextContent | ImageContent)[] = [];
	for (const part of parts) {
		if (part.type === 'text') {
			blocks.push({ type: 'text', text: part.content });
			continue;
		}
		if (part.type === 'document' && part.source.type === 'data') {
			const filename = isJsonObject(part.metadata) ? part.metadata.filename : undefined;
			const document: DocumentContextBlock = {
				type: 'image',
				data: part.source.value,
				mimeType: part.source.mimeType,
				...(typeof filename === 'string' ? { filename } : {}),
			};
			blocks.push(document);
		} else if (part.type === 'image' && part.source.type === 'data') {
			blocks.push({
				type: 'image',
				data: part.source.value,
				mimeType: part.source.mimeType,
			});
		} else {
			// ponytail: Flue only sends inline images and documents; other media reads as a marker.
			blocks.push({ type: 'text', text: `[${part.type}]` });
		}
	}
	return blocks;
}

function fromAssistantModelMessage(message: ModelMessage, timestamp: number) {
	const metadata = tanstackMetadata(message.metadata);
	const text =
		typeof message.content === 'string'
			? message.content
			: (message.content ?? [])
					.flatMap((part) => (part.type === 'text' ? [part.content] : []))
					.join('');
	const blocks: OrderedAssistantBlock[] = orderedAssistantBlocks(message) ?? [
		...(message.thinking ?? []).map((thinking) => ({
			type: 'thinking' as const,
			thinking,
		})),
		...(text ? [{ type: 'text' as const, text }] : []),
		...(message.toolCalls ?? []).map((toolCall) => ({
			type: 'tool-call' as const,
			toolCall,
		})),
	];
	const content = blocks.map(fromOrderedBlock);
	const hasToolCalls = content.some((block) => block.type === 'toolCall');
	return {
		role: 'assistant',
		content,
		api: metadata?.source?.api ?? '',
		provider: metadata?.source?.provider ?? '',
		model: metadata?.source?.model ?? '',
		usage: toFlueUsage(undefined, undefined),
		stopReason: metadata?.stopReason ?? (hasToolCalls ? 'toolUse' : 'stop'),
		timestamp,
	} satisfies AssistantMessage;
}

function fromOrderedBlock(block: OrderedAssistantBlock) {
	switch (block.type) {
		case 'thinking':
			return {
				type: 'thinking',
				thinking: block.thinking.redacted ? REDACTED_THINKING_TEXT : block.thinking.content,
				...(block.thinking.signature ? { thinkingSignature: block.thinking.signature } : {}),
				...(block.thinking.redacted ? { redacted: true } : {}),
			} satisfies ThinkingContent;
		case 'text':
			return { type: 'text', text: block.text } satisfies TextContent;
		case 'tool-call': {
			const thoughtSignature = isJsonObject(block.toolCall.metadata)
				? block.toolCall.metadata.thoughtSignature
				: undefined;
			return {
				type: 'toolCall',
				id: block.toolCall.id,
				name: block.toolCall.function.name,
				arguments: parseToolArguments(block.toolCall.function.arguments, undefined),
				...(typeof thoughtSignature === 'string' ? { thoughtSignature } : {}),
			} satisfies ToolCall;
		}
	}
}

/** A block event of a streaming assistant message, in pi's shape. */
export type AssistantBlockEvent =
	| { type: 'text_start'; contentIndex: number; partial: AssistantMessage }
	| {
			type: 'text_delta';
			contentIndex: number;
			delta: string;
			partial: AssistantMessage;
	  }
	| {
			type: 'text_end';
			contentIndex: number;
			content: string;
			partial: AssistantMessage;
	  }
	| { type: 'thinking_start'; contentIndex: number; partial: AssistantMessage }
	| {
			type: 'thinking_delta';
			contentIndex: number;
			delta: string;
			partial: AssistantMessage;
	  }
	| {
			type: 'thinking_end';
			contentIndex: number;
			content: string;
			partial: AssistantMessage;
	  }
	| {
			type: 'toolcall_delta';
			contentIndex: number;
			delta: string;
			partial: AssistantMessage;
	  }
	| {
			type: 'toolcall_end';
			contentIndex: number;
			toolCall: ToolCall;
			partial: AssistantMessage;
	  };

/**
 * Builds a Flue assistant message from the TanStack stream chunks of one
 * model call. `push` returns the block events of each chunk, and `finish`
 * closes the open blocks and returns the message.
 *
 * A thinking block ends at the next block or at the end of the run, not at
 * its own end event: OpenAI sends the reasoning signature after that event.
 */
export class AssistantStreamAssembler {
	private readonly message: AssistantMessage;
	private openTextIndex: number | undefined;
	private openThinkingIndex: number | undefined;
	private endedThinkingIndex: number | undefined;
	/** Reasoning message ids and step ids to the content index of their thinking block. */
	private readonly thinkingIndexes = new Map<string, number>();
	private readonly toolCalls = new Map<string, { index: number; args: string }>();

	constructor(private readonly info: FlueModelInfo) {
		this.message = {
			role: 'assistant',
			content: [],
			api: info.api,
			provider: info.provider,
			model: info.id,
			usage: toFlueUsage(undefined, info.cost),
			stopReason: 'stop',
			timestamp: Date.now(),
		};
	}

	/** A copy of the message so far. */
	snapshot() {
		return structuredClone(this.message);
	}

	push(chunk: StreamChunk) {
		const events: AssistantBlockEvent[] = [];
		switch (chunk.type) {
			case EventType.TEXT_MESSAGE_START:
				if (this.openTextIndex === undefined) this.startText(events);
				break;
			case EventType.TEXT_MESSAGE_CONTENT: {
				const index = this.openTextIndex ?? this.startText(events);
				const block = this.blockAt(index, 'text');
				block.text += chunk.delta;
				events.push({
					type: 'text_delta',
					contentIndex: index,
					delta: chunk.delta,
					partial: this.message,
				});
				break;
			}
			case EventType.TEXT_MESSAGE_END:
				this.closeText(events);
				break;
			case EventType.REASONING_START:
			case EventType.REASONING_MESSAGE_START:
				this.thinkingIndexFor(chunk.messageId, events);
				break;
			case EventType.STEP_STARTED: {
				const index = this.openThinkingIndex;
				if (index !== undefined) this.thinkingIndexes.set(stepKey(chunk), index);
				break;
			}
			case EventType.REASONING_MESSAGE_CONTENT: {
				const index = this.thinkingIndexFor(chunk.messageId, events);
				const block = this.blockAt(index, 'thinking');
				if (block.redacted) break;
				block.thinking += chunk.delta;
				events.push({
					type: 'thinking_delta',
					contentIndex: index,
					delta: chunk.delta,
					partial: this.message,
				});
				break;
			}
			case EventType.STEP_FINISHED: {
				const signature = 'signature' in chunk ? chunk.signature : undefined;
				const index = this.thinkingIndexes.get(stepKey(chunk)) ?? this.lastThinkingIndex();
				if (typeof signature !== 'string' || !signature || index === undefined) break;
				const block = this.blockAt(index, 'thinking');
				if (!block.redacted) block.thinkingSignature = signature;
				break;
			}
			case EventType.REASONING_ENCRYPTED_VALUE:
				if (chunk.subtype === 'tool-call') {
					const call = this.toolCalls.get(chunk.entityId);
					if (call) this.blockAt(call.index, 'toolCall').thoughtSignature = chunk.encryptedValue;
				} else {
					const block = this.blockAt(this.thinkingIndexFor(chunk.entityId, events), 'thinking');
					block.thinkingSignature = chunk.encryptedValue;
				}
				break;
			case EventType.REASONING_MESSAGE_END:
			case EventType.REASONING_END:
				if (this.openThinkingIndex !== undefined) this.endedThinkingIndex = this.openThinkingIndex;
				this.openThinkingIndex = undefined;
				break;
			case EventType.TOOL_CALL_START: {
				this.closeBlocks(events);
				const thoughtSignature = chunk.metadata?.thoughtSignature;
				this.message.content.push({
					type: 'toolCall',
					id: chunk.toolCallId,
					name: chunk.toolCallName,
					arguments: {},
					...(typeof thoughtSignature === 'string' && thoughtSignature ? { thoughtSignature } : {}),
				});
				this.toolCalls.set(chunk.toolCallId, {
					index: this.message.content.length - 1,
					args: '',
				});
				break;
			}
			case EventType.TOOL_CALL_ARGS: {
				const call = this.toolCalls.get(chunk.toolCallId);
				if (!call) break;
				call.args += chunk.delta;
				events.push({
					type: 'toolcall_delta',
					contentIndex: call.index,
					delta: chunk.delta,
					partial: this.message,
				});
				break;
			}
			case EventType.TOOL_CALL_END: {
				const call = this.toolCalls.get(chunk.toolCallId);
				if (!call) break;
				const block = this.blockAt(call.index, 'toolCall');
				block.arguments = parseToolArguments(call.args, chunk.input);
				events.push({
					type: 'toolcall_end',
					contentIndex: call.index,
					toolCall: block,
					partial: this.message,
				});
				break;
			}
			case EventType.RUN_FINISHED:
				this.closeBlocks(events);
				this.finishRun(chunk.finishReason ?? tanstackMetadata(chunk)?.finishReason ?? null);
				this.setUsage(chunk);
				if (chunk.responseId) this.message.responseId = chunk.responseId;
				if (chunk.model && chunk.model !== this.info.id) this.message.responseModel = chunk.model;
				this.attachDiagnostics(chunk);
				break;
			case EventType.RUN_ERROR: {
				this.closeBlocks(events);
				const isAborted =
					chunk.code === 'aborted' || tanstackMetadata(chunk)?.stopReason === 'aborted';
				this.message.stopReason = isAborted ? 'aborted' : 'error';
				this.message.errorMessage = chunk.message || chunk.error?.message || 'Run failed';
				this.setUsage(chunk);
				this.attachDiagnostics(chunk);
				break;
			}
		}
		return events;
	}

	/** Provider-response metadata that the adapter put on the chunk. */
	private attachDiagnostics(chunk: StreamChunk) {
		const diagnostics = chunkProviderResponseDiagnostics(chunk);
		if (diagnostics) attachProviderResponseDiagnostics(this.message, diagnostics);
	}

	/** Close the open blocks and return the message. Call it once, after the last chunk. */
	finish() {
		this.closeBlocks([]);
		return this.message;
	}

	private startText(events: AssistantBlockEvent[]) {
		this.closeBlocks(events);
		this.message.content.push({ type: 'text', text: '' });
		const index = this.message.content.length - 1;
		this.openTextIndex = index;
		events.push({
			type: 'text_start',
			contentIndex: index,
			partial: this.message,
		});
		return index;
	}

	private closeText(events: AssistantBlockEvent[]) {
		const index = this.openTextIndex;
		if (index === undefined) return;
		this.openTextIndex = undefined;
		const content = this.blockAt(index, 'text').text;
		events.push({
			type: 'text_end',
			contentIndex: index,
			content,
			partial: this.message,
		});
	}

	/** End every open block: the thinking blocks first, then the text. */
	private closeBlocks(events: AssistantBlockEvent[]) {
		for (const index of [this.endedThinkingIndex, this.openThinkingIndex]) {
			if (index === undefined) continue;
			const content = this.blockAt(index, 'thinking').thinking;
			events.push({
				type: 'thinking_end',
				contentIndex: index,
				content,
				partial: this.message,
			});
		}
		this.endedThinkingIndex = undefined;
		this.openThinkingIndex = undefined;
		this.closeText(events);
	}

	/** The thinking block of a reasoning id. An unknown id starts a new block. */
	private thinkingIndexFor(id: string, events: AssistantBlockEvent[]) {
		const known = this.thinkingIndexes.get(id);
		if (known !== undefined) return known;
		this.closeBlocks(events);
		const redacted = id.startsWith(REDACTED_THINKING_ID_PREFIX);
		this.message.content.push({
			type: 'thinking',
			thinking: redacted ? REDACTED_THINKING_TEXT : '',
			...(redacted ? { redacted: true } : {}),
		});
		const index = this.message.content.length - 1;
		this.thinkingIndexes.set(id, index);
		this.openThinkingIndex = index;
		events.push({
			type: 'thinking_start',
			contentIndex: index,
			partial: this.message,
		});
		return index;
	}

	private lastThinkingIndex() {
		return this.openThinkingIndex ?? this.endedThinkingIndex;
	}

	private blockAt<TType extends AssistantMessage['content'][number]['type']>(
		index: number,
		type: TType,
	) {
		const block = this.message.content[index];
		if (!isBlockOfType(block, type))
			throw new Error(`[flue] Expected a ${type} block at index ${index}.`);
		return block;
	}

	private finishRun(finishReason: 'stop' | 'length' | 'content_filter' | 'tool_calls' | null) {
		switch (finishReason) {
			case 'tool_calls':
				this.message.stopReason = 'toolUse';
				break;
			case 'length':
				this.message.stopReason = 'length';
				break;
			case 'content_filter':
				this.message.stopReason = 'error';
				this.message.errorMessage = 'Provider finish_reason: content_filter';
				break;
			case 'stop':
			case null:
				this.message.stopReason = 'stop';
				break;
		}
		if (finishReason) this.message.rawStopReason = finishReason;
	}

	private setUsage(chunk: RunFinishedEvent | RunErrorEvent) {
		const usage = Array.isArray(chunk.usage)
			? fromSpecTokenUsage(chunk.usage, tanstackMetadata(chunk)?.usage)
			: chunk.usage;
		if (usage) this.message.usage = toFlueUsage(usage, this.info.cost);
	}
}

function isBlockOfType<TType extends AssistantMessage['content'][number]['type']>(
	block: AssistantMessage['content'][number] | undefined,
	type: TType,
): block is Extract<AssistantMessage['content'][number], { type: TType }> {
	return block?.type === type;
}

function stepKey(chunk: { stepName: string }) {
	return 'stepId' in chunk && typeof chunk.stepId === 'string' ? chunk.stepId : chunk.stepName;
}

function parseToolArguments(raw: string, input: unknown): JsonObject {
	if (isJsonObject(input)) return input;
	try {
		const parsed: unknown = JSON.parse(raw || '{}');
		return isJsonObject(parsed) ? parsed : {};
	} catch {
		// pi kept what it could parse of cut arguments; an unparsable object becomes empty.
		return {};
	}
}

function isJsonObject(value: unknown): value is JsonObject {
	return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * TanStack usage in Flue's shape. TanStack counts cached tokens inside
 * `promptTokens`; Flue's `input` is the uncached part, as pi's was.
 */
export function toFlueUsage(
	usage: TokenUsage | undefined,
	rates: ModelCostRates | undefined,
): Usage {
	const cacheRead = usage?.promptTokensDetails?.cachedTokens ?? 0;
	const cacheWrite = usage?.promptTokensDetails?.cacheWriteTokens ?? 0;
	const cacheWrite1h = usage?.promptTokensDetails?.cacheWrite1hTokens;
	const input = Math.max(0, (usage?.promptTokens ?? 0) - cacheRead - cacheWrite);
	const output = usage?.completionTokens ?? 0;
	const reasoning = usage?.completionTokensDetails?.reasoningTokens;
	const counts = {
		input,
		output,
		cacheRead,
		cacheWrite,
		...(cacheWrite1h === undefined ? {} : { cacheWrite1h }),
	};
	return {
		...counts,
		...(reasoning === undefined ? {} : { reasoning }),
		totalTokens: usage?.totalTokens ?? 0,
		cost: rates
			? modelCost({ cost: rates }, counts)
			: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}
