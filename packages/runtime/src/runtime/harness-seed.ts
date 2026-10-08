/**
 * The starting harness transcript of a conversation that Flue wrote before
 * the harness ran it (an old stream).
 *
 * An old stream has only Flue records, and the host `project` fold adds only
 * signals to the transcript. So the harness transcript of an old thread has
 * no history. At the first open of such a thread, the session appends one
 * seed record: Flue's model context, converted to TanStack messages as the
 * durable path stores them. The `project` fold makes the seed the
 * transcript (see `projectFlueRecord`).
 */
import type { MessageSource, ModelMessage } from '@tanstack/ai';
import { tanstackMetadata } from '@tanstack/ai/adapter-internals';
import type { LogRecord } from '@tanstack/ai-persistence';
import type { AgentMessage, ImageContent, TextContent } from '../llm-types.ts';
import { type FlueModelInfo, modelInfo, toModelRequest, toUserInput } from '../model-messages.ts';
import type { FlueModel } from '../providers/provider.ts';

/** The type of the seed record. It is a host record, not a Flue conversation record. */
const FLUE_SEED_RECORD_TYPE = 'flue.seed';

/**
 * The transcript that the durable path stores for Flue's model context
 * `messages` (what `buildConversationContext` gives):
 *
 * - A user or signal message is the user input that the harness gets: one
 *   text block as plain text, other content as parts.
 * - An assistant message is the message that a model call gives back, as
 *   Flue sends it to a model. A message of `model` names `source` as its
 *   source, as the model call of the durable path does: the TanStack
 *   identity of the model's adapter. That keeps its replay the same (a
 *   foreign source turns reasoning into text and rewrites tool call ids).
 *   The context holds a failed answer only as a cut answer before the two
 *   stream recovery signals. The durable path keeps a cut answer in the
 *   transcript as an answer, so the model reads it, and so does the seed:
 *   it has no failed stop reason (TanStack drops a failed answer).
 * - A tool result is the tool message of the tool bridge: one text block as
 *   its exact text, other content as parts.
 *
 * The system message is left out: the prompt goes in each model call.
 * `model` is the model of the conversation.
 *
 * @example
 * ```ts
 * const source = { provider: adapter.provider ?? adapter.name, api: adapter.api ?? adapter.kind, model: model.id };
 * const messages = seedTranscriptOf(agentLoop.state.messages, model, source);
 * ```
 */
export function seedTranscriptOf(
	messages: readonly AgentMessage[],
	model: FlueModel,
	source: MessageSource,
) {
	const target = modelInfo(model);
	return messages.flatMap<ModelMessage>((message) =>
		toSeedMessages(message, model, target, source),
	);
}

/** The transcript messages of one message of Flue's context. See {@link seedTranscriptOf}. */
function toSeedMessages(
	message: AgentMessage,
	model: FlueModel,
	target: FlueModelInfo,
	source: MessageSource,
) {
	switch (message.role) {
		case 'system':
			return [];
		case 'user':
		case 'signal':
			return [{ role: 'user', content: toUserInput(message) } satisfies ModelMessage];
		case 'assistant': {
			const converted = toModelRequest(
				{ systemPrompt: '', messages: [message], tools: [] },
				target,
			).messages;
			const isOfModel =
				message.provider === model.provider &&
				message.api === model.api &&
				message.model === model.id;
			return converted.map((item) => asAnswer(item, isOfModel ? source : undefined));
		}
		case 'toolResult':
			return [
				{
					role: 'tool',
					content: toolMessageContent(message.content),
					toolCallId: message.toolCallId,
				} satisfies ModelMessage,
			];
	}
}

/** `message` with no failed stop reason, and with `source` as its TanStack source when set. */
function asAnswer(message: ModelMessage, source: MessageSource | undefined) {
	const { stopReason: _stopReason, ...tanstack } = tanstackMetadata(message.metadata) ?? {};
	return {
		...message,
		metadata: { tanstack: { ...tanstack, ...(source ? { source } : {}) } },
	} satisfies ModelMessage;
}

/** The tool message content of a tool's result, as the tool bridge gives it to TanStack. */
function toolMessageContent(content: readonly (TextContent | ImageContent)[]) {
	const [only] = content;
	if (content.length === 1 && only?.type === 'text') return only.text;
	return content.map((block) =>
		block.type === 'text'
			? { type: 'text' as const, content: block.text }
			: {
					type: 'image' as const,
					source: { type: 'data' as const, value: block.data, mimeType: block.mimeType },
				},
	);
}

/**
 * The seed record of thread `thread`. Append it with `session.append` of
 * that thread.
 */
export function flueSeedRecord(thread: string, messages: readonly ModelMessage[]) {
	return { type: FLUE_SEED_RECORD_TYPE, thread, messages: [...messages] } satisfies LogRecord;
}

/** The messages of a seed record, or `undefined` for another record. */
export function seedMessagesOf(record: LogRecord) {
	if (record.type !== FLUE_SEED_RECORD_TYPE) return undefined;
	const { messages } = record;
	if (!Array.isArray(messages) || !messages.every(isModelMessage)) return undefined;
	return messages;
}

function isModelMessage(value: unknown): value is ModelMessage {
	return (
		typeof value === 'object' &&
		value !== null &&
		'role' in value &&
		(value.role === 'user' || value.role === 'assistant' || value.role === 'tool')
	);
}

/**
 * True when no message of the harness transcript came from the harness: each
 * one is a message that the `project` fold made from a Flue signal record.
 * Such a message has the signal's message id, and `signalIds` holds the
 * message ids of the conversation's signals. The harness gives a message it
 * writes its own id or none, and a seed message has none.
 */
export function holdsOnlyProjectedSignals(
	transcript: readonly ModelMessage[],
	signalIds: ReadonlySet<string>,
) {
	return transcript.every(
		(message) =>
			message.role === 'user' && typeof message.id === 'string' && signalIds.has(message.id),
	);
}

/**
 * True when `seed` is the transcript `transcript` already holds, the message
 * ids aside. The seed of a thread with only signals is the transcript that
 * the `project` fold built.
 */
export function sameTranscript(transcript: readonly ModelMessage[], seed: readonly ModelMessage[]) {
	return (
		transcript.length === seed.length &&
		transcript.every((message, index) => {
			const { id: _id, ...rest } = message;
			return JSON.stringify(rest) === JSON.stringify(seed[index]);
		})
	);
}
