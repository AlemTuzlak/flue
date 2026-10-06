/**
 * Context compaction for long sessions. When context approaches the model's
 * window limit, older messages are summarized and replaced with a structured summary.
 *
 * Trigger modes:
 * 1. Threshold — tokens exceed (contextWindow - reserveTokens). Compact, no retry.
 * 2. Overflow — LLM returned context overflow. Compact, then auto-retry.
 */
import {
	type AnyTextAdapter,
	isContextOverflow,
	type ModelMessage,
	type SystemPrompt,
} from '@tanstack/ai';
import { conversationSummarizer } from '@tanstack/ai-compaction';
import { WORKERS_AI_OVERFLOW_MARKER } from './errors.ts';
import type {
	AgentMessage,
	AssistantMessage,
	ToolResultMessage,
	Usage,
	UserMessage,
} from './llm-types.ts';
import { AssistantStreamAssembler, modelInfo, toModelRequest } from './model-messages.ts';
import { outputTokenOptions } from './providers/adapters.ts';
import type { FlueModel } from './providers/provider.ts';
import type { PromptUsage } from './types.ts';
import { addUsage, fromProviderUsage } from './usage.ts';

// ─── Settings ───────────────────────────────────────────────────────────────

export interface CompactionSettings {
	enabled: boolean;
	reserveTokens: number;
	keepRecentTokens: number;
}

/**
 * Defaults applied when no user config and no model metadata are available.
 * Real sessions construct settings via {@link deriveCompactionDefaults} so
 * headroom tracks the active model instead of a fixed Sonnet-sized window.
 */
export const DEFAULT_COMPACTION_SETTINGS: CompactionSettings = {
	enabled: true,
	reserveTokens: 20000,
	keepRecentTokens: 8000,
};

/**
 * Compute model-aware defaults. Reserve is capped at the model's max output
 * because reserving more than the model can emit in one turn wastes context;
 * the preserved tail stays flat because recent-context fidelity depends on
 * the active work, not on the model's total window size.
 *
 * Caller may override either field after calling this.
 */
export function deriveCompactionDefaults(input: {
	contextWindow: number;
	maxTokens: number;
}): CompactionSettings {
	// When `maxTokens` is unknown (e.g. HTTP providers without declared
	// metadata), fall back to the static reserve.
	const reserveCap =
		input.maxTokens > 0 ? input.maxTokens : DEFAULT_COMPACTION_SETTINGS.reserveTokens;
	let reserveTokens = Math.min(DEFAULT_COMPACTION_SETTINGS.reserveTokens, reserveCap);
	// Safety floor for tiny-window models: reserve must leave room for at
	// least some meaningful context. If reserve would consume half or more
	// of the window, clamp to a third of the window so threshold compaction
	// can actually fire usefully instead of triggering on every turn.
	if (input.contextWindow > 0 && reserveTokens * 2 >= input.contextWindow) {
		reserveTokens = Math.max(1024, Math.floor(input.contextWindow / 3));
	}
	return {
		enabled: true,
		reserveTokens,
		keepRecentTokens: DEFAULT_COMPACTION_SETTINGS.keepRecentTokens,
	};
}

// ─── Token Estimation ───────────────────────────────────────────────────────

export function calculateContextTokens(usage: Usage): number {
	return usage.totalTokens || usage.input + usage.output + usage.cacheRead + usage.cacheWrite;
}

function getAssistantUsage(msg: AgentMessage): Usage | undefined {
	if (msg.role === 'assistant' && 'usage' in msg) {
		const assistantMsg = msg as AssistantMessage;
		if (
			assistantMsg.stopReason !== 'aborted' &&
			assistantMsg.stopReason !== 'error' &&
			assistantMsg.usage
		) {
			return assistantMsg.usage;
		}
	}
	return undefined;
}

function getLastAssistantUsageInfo(
	messages: AgentMessage[],
): { usage: Usage; index: number } | undefined {
	for (let i = messages.length - 1; i >= 0; i--) {
		const msg = messages[i];
		if (!msg) continue;
		const usage = getAssistantUsage(msg);
		if (usage) return { usage, index: i };
	}
	return undefined;
}

/** chars/4 heuristic. Conservative (overestimates). */
function estimateTokens(message: AgentMessage): number {
	let chars = 0;
	switch (message.role) {
		case 'user': {
			const { content } = message as UserMessage;
			if (typeof content === 'string') {
				chars = content.length;
			} else if (Array.isArray(content)) {
				for (const block of content) {
					if (block.type === 'text') {
						chars += block.text.length;
					}
				}
			}
			return Math.ceil(chars / 4);
		}
		case 'assistant': {
			const { content } = message as AssistantMessage;
			for (const block of content) {
				if (block.type === 'text') {
					chars += block.text.length;
				} else if (block.type === 'thinking') {
					chars += block.thinking.length;
				} else if (block.type === 'toolCall') {
					chars += block.name.length + JSON.stringify(block.arguments).length;
				}
			}
			return Math.ceil(chars / 4);
		}
		case 'toolResult': {
			const { content } = message as ToolResultMessage;
			for (const block of content) {
				if (block.type === 'text') {
					chars += block.text.length;
				} else if (block.type === 'image') {
					// Approximate token cost for an image block
					chars += 4800;
				}
			}
			return Math.ceil(chars / 4);
		}
	}
	return 0;
}

/** pi's estimate: the last assistant's reported usage, plus chars/4 for each later message. */
export function estimateContextTokens(messages: AgentMessage[]): number {
	const usageInfo = getLastAssistantUsageInfo(messages);
	if (!usageInfo) {
		let estimated = 0;
		for (const message of messages) {
			estimated += estimateTokens(message);
		}
		return estimated;
	}
	const usageTokens = calculateContextTokens(usageInfo.usage);
	let trailingTokens = 0;
	for (let i = usageInfo.index + 1; i < messages.length; i++) {
		const message = messages[i];
		if (message) trailingTokens += estimateTokens(message);
	}
	return usageTokens + trailingTokens;
}

export function shouldCompact(
	contextTokens: number,
	contextWindow: number,
	settings: CompactionSettings,
): boolean {
	if (!settings.enabled) return false;
	// `contextWindow <= 0` means unknown — skip threshold; overflow recovery still runs.
	if (contextWindow <= 0) return false;
	return contextTokens > contextWindow - settings.reserveTokens;
}

// ─── File Operation Tracking ────────────────────────────────────────────────

interface FileOps {
	read: Set<string>;
	written: Set<string>;
	edited: Set<string>;
}

function createFileOps(): FileOps {
	return { read: new Set(), written: new Set(), edited: new Set() };
}

function extractFileOpsFromMessage(message: AgentMessage, fileOps: FileOps): void {
	if (message.role !== 'assistant') return;
	const assistant = message as AssistantMessage;
	if (!Array.isArray(assistant.content)) return;
	for (const block of assistant.content) {
		if (block.type !== 'toolCall') continue;
		const args = block.arguments;
		if (!args) continue;
		const path = typeof args.path === 'string' ? args.path : undefined;
		if (!path) continue;
		switch (block.name) {
			case 'read':
				fileOps.read.add(path);
				break;
			case 'write':
				fileOps.written.add(path);
				break;
			case 'edit':
				fileOps.edited.add(path);
				break;
		}
	}
}

function computeFileLists(fileOps: FileOps): {
	readFiles: string[];
	modifiedFiles: string[];
} {
	const modified = new Set([...fileOps.edited, ...fileOps.written]);
	const readOnly = [...fileOps.read].filter((f) => !modified.has(f)).sort();
	const modifiedFiles = [...modified].sort();
	return { readFiles: readOnly, modifiedFiles };
}

function formatFileOperations(readFiles: string[], modifiedFiles: string[]): string {
	const sections: string[] = [];
	if (readFiles.length > 0) {
		sections.push(`<read-files>\n${readFiles.join('\n')}\n</read-files>`);
	}
	if (modifiedFiles.length > 0) {
		sections.push(`<modified-files>\n${modifiedFiles.join('\n')}\n</modified-files>`);
	}
	if (sections.length === 0) return '';
	return `\n\n${sections.join('\n\n')}`;
}

// ─── Cut Point Detection ────────────────────────────────────────────────────

/** Valid cut points: user or assistant messages. Never cut at toolResult. */
function findValidCutPoints(messages: AgentMessage[], start: number, end: number): number[] {
	const cutPoints: number[] = [];
	for (let i = start; i < end; i++) {
		const role = messages[i]?.role;
		if (role === 'user' || role === 'assistant') {
			cutPoints.push(i);
		}
	}
	return cutPoints;
}

function findTurnStartIndex(messages: AgentMessage[], index: number, start: number): number {
	for (let i = index; i >= start; i--) {
		if (messages[i]?.role === 'user') return i;
	}
	return -1;
}

interface CutPointResult {
	firstKeptIndex: number;
	turnStartIndex: number;
	isSplitTurn: boolean;
}

function findCutPoint(
	messages: AgentMessage[],
	start: number,
	end: number,
	keepRecentTokens: number,
): CutPointResult {
	const cutPoints = findValidCutPoints(messages, start, end);
	if (cutPoints.length === 0) {
		return { firstKeptIndex: start, turnStartIndex: -1, isSplitTurn: false };
	}

	let accumulatedTokens = 0;
	let cutIndex = cutPoints[0] ?? start;

	for (let i = end - 1; i >= start; i--) {
		const message = messages[i];
		if (!message) continue;
		const messageTokens = estimateTokens(message);
		accumulatedTokens += messageTokens;
		if (accumulatedTokens >= keepRecentTokens) {
			for (const cutPoint of cutPoints) {
				if (cutPoint >= i) {
					cutIndex = cutPoint;
					break;
				}
			}
			break;
		}
	}

	const isUserMessage = messages[cutIndex]?.role === 'user';
	const turnStartIndex = isUserMessage ? -1 : findTurnStartIndex(messages, cutIndex, start);

	return {
		firstKeptIndex: cutIndex,
		turnStartIndex,
		isSplitTurn: !isUserMessage && turnStartIndex !== -1,
	};
}

// ─── Compaction Preparation ─────────────────────────────────────────────────

export interface CompactionPreparation {
	firstKeptIndex: number;
	messagesToSummarize: AgentMessage[];
	turnPrefixMessages: AgentMessage[];
	isSplitTurn: boolean;
	tokensBefore: number;
	previousSummary: string | undefined;
	fileOps: FileOps;
	settings: CompactionSettings;
}

export interface CompactionResult {
	summary: string;
	tokensBefore: number;
	details: { readFiles: string[]; modifiedFiles: string[] };
	/**
	 * Aggregate token usage from the 1–2 summarization calls that produced
	 * this result. Undefined when no call reported usage (rare — some
	 * providers may stream without totals). Already normalized into Flue's
	 * `PromptUsage` shape so callers can persist it directly on a
	 * `CompactionEntry`.
	 */
	usage?: PromptUsage;
}

export interface CompactionTurnHandle {
	turnId: string;
}

export interface CompactionTurnObserver {
	start(
		purpose: 'compaction' | 'compaction_prefix',
		model: FlueModel,
		context: {
			systemPrompt: string;
			messages: AgentMessage[];
			tools: [];
		},
		options: { maxTokens: number },
	): CompactionTurnHandle;
	run<T>(handle: CompactionTurnHandle, execute: () => Promise<T>): Promise<T>;
	end(
		purpose: 'compaction' | 'compaction_prefix',
		handle: CompactionTurnHandle,
		response: AssistantMessage | undefined,
		error: unknown | undefined,
	): void;
}

/** Pure function — no I/O. Finds cut point, extracts messages to summarize, tracks file ops. */
export function prepareCompaction(
	messages: AgentMessage[],
	settings: CompactionSettings,
	previousCompaction?: {
		summary: string;
		firstKeptIndex: number;
		details?: { readFiles: string[]; modifiedFiles: string[] };
	},
): CompactionPreparation | undefined {
	if (messages.length === 0) return undefined;

	const boundaryStart = previousCompaction ? previousCompaction.firstKeptIndex : 0;
	const boundaryEnd = messages.length;
	const tokensBefore = estimateContextTokens(messages);

	const cutPoint = findCutPoint(messages, boundaryStart, boundaryEnd, settings.keepRecentTokens);

	if (cutPoint.firstKeptIndex <= boundaryStart) return undefined;

	const historyEnd = cutPoint.isSplitTurn ? cutPoint.turnStartIndex : cutPoint.firstKeptIndex;

	const messagesToSummarize = messages.slice(boundaryStart, historyEnd);
	const turnPrefixMessages = cutPoint.isSplitTurn
		? messages.slice(cutPoint.turnStartIndex, cutPoint.firstKeptIndex)
		: [];

	const fileOps = createFileOps();
	if (previousCompaction?.details) {
		for (const f of previousCompaction.details.readFiles ?? []) fileOps.read.add(f);
		for (const f of previousCompaction.details.modifiedFiles ?? []) fileOps.edited.add(f);
	}
	for (const msg of messagesToSummarize) {
		extractFileOpsFromMessage(msg, fileOps);
	}
	for (const msg of turnPrefixMessages) {
		extractFileOpsFromMessage(msg, fileOps);
	}

	return {
		firstKeptIndex: cutPoint.firstKeptIndex,
		messagesToSummarize,
		turnPrefixMessages,
		isSplitTurn: cutPoint.isSplitTurn,
		tokensBefore,
		previousSummary: previousCompaction?.summary,
		fileOps,
		settings,
	};
}

// ─── Summary Generation ─────────────────────────────────────────────────────

/** The text of a TanStack system prompt. */
function systemPromptText(prompt: SystemPrompt) {
	return typeof prompt === 'string' ? prompt : prompt.content;
}

/** A TanStack request message as a Flue message, for the request observation. */
function observedMessage(message: ModelMessage): AgentMessage {
	const text =
		typeof message.content === 'string'
			? message.content
			: (message.content ?? []).map((part) => (part.type === 'text' ? part.content : '')).join('');
	return {
		role: 'user',
		content: [{ type: 'text', text }],
		timestamp: Date.now(),
	};
}

/**
 * `adapter` with the observer around its one summary call: the request
 * observation before it, each stream step inside `observer.run`, and the
 * assembled answer after it.
 */
function observedAdapter(
	adapter: AnyTextAdapter,
	model: FlueModel,
	purpose: 'compaction' | 'compaction_prefix',
	maxTokens: number,
	observer: CompactionTurnObserver,
	onResponse: (response: AssistantMessage) => void,
) {
	const chatStream: AnyTextAdapter['chatStream'] = async function* (options) {
		const handle = observer.start(
			purpose,
			model,
			{
				systemPrompt: options.systemPrompts?.map(systemPromptText).join('\n') ?? '',
				messages: options.messages.map(observedMessage),
				tools: [],
			},
			{ maxTokens },
		);
		const assembler = new AssistantStreamAssembler(modelInfo(model));
		try {
			const iterator = adapter.chatStream(options)[Symbol.asyncIterator]();
			while (true) {
				const step = await observer.run(handle, () => iterator.next());
				if (step.done) break;
				assembler.push(step.value);
				yield step.value;
			}
			const response = assembler.finish();
			onResponse(response);
			observer.end(purpose, handle, response, undefined);
		} catch (error) {
			observer.end(purpose, handle, undefined, error);
			throw error;
		}
	};
	const observed: AnyTextAdapter = Object.create(adapter);
	return Object.assign(observed, { chatStream });
}

/** One summary call with TanStack's summary prompt, observed as a Flue model turn. */
async function summarize(
	purpose: 'compaction' | 'compaction_prefix',
	messages: AgentMessage[],
	input: { previousSummary?: string; turnPrefix?: boolean },
	maxTokens: number,
	target: { model: FlueModel; adapter: AnyTextAdapter },
	signal: AbortSignal,
	observer: CompactionTurnObserver,
	errorPrefix: string,
): Promise<{ text: string; usage: Usage | undefined }> {
	const { model } = target;
	let response: AssistantMessage | undefined;
	const summarizer = conversationSummarizer({
		adapter: observedAdapter(target.adapter, model, purpose, maxTokens, observer, (answer) => {
			response = answer;
		}),
		modelOptions: outputTokenOptions(model, maxTokens),
	});
	const request = toModelRequest({ systemPrompt: '', messages, tools: [] }, modelInfo(model));
	const result = await summarizer(request.messages, { ...input, signal });
	if (response?.stopReason === 'error') {
		throw new Error(`${errorPrefix}: ${response.errorMessage || 'Unknown error'}`);
	}
	const text = typeof result === 'string' ? result : result.summary;
	return { text, usage: response?.usage };
}

// ─── Main Compaction Function ───────────────────────────────────────────────

export async function compact(
	preparation: CompactionPreparation,
	target: { model: FlueModel; adapter: AnyTextAdapter },
	signal: AbortSignal,
	observer: CompactionTurnObserver,
): Promise<CompactionResult> {
	const {
		messagesToSummarize,
		turnPrefixMessages,
		isSplitTurn,
		tokensBefore,
		previousSummary,
		fileOps,
		settings,
	} = preparation;

	let summary: string;
	// Sum the usage of every summarization call that produced a value.
	// Split-turn compaction fires two calls; regular compaction fires one.
	// A call may report `undefined` usage (rare provider behaviour) — those
	// contribute zero. Normalize the message `Usage` to Flue's `PromptUsage`
	// at this boundary so the result is a persistable shape for the
	// downstream `CompactionEntry`.
	let aggregateUsage: PromptUsage | undefined;
	const addCallUsage = (usage: Usage | undefined): void => {
		const normalized = fromProviderUsage(usage);
		if (!normalized) return;
		aggregateUsage = aggregateUsage ? addUsage(aggregateUsage, normalized) : normalized;
	};

	// pi's caps for the summary calls: 80% of the reserve for the history,
	// 50% for a split turn's prefix, at most 16k tokens each.
	const historyTokens = Math.min(Math.floor(0.8 * settings.reserveTokens), 16_000);
	const prefixTokens = Math.min(Math.floor(0.5 * settings.reserveTokens), 16_000);
	const summarizeHistory = () =>
		summarize(
			'compaction',
			messagesToSummarize,
			previousSummary ? { previousSummary } : {},
			historyTokens,
			target,
			signal,
			observer,
			'Summarization failed',
		);

	if (isSplitTurn && turnPrefixMessages.length > 0) {
		const [historyResult, turnPrefixResult] = await Promise.all([
			messagesToSummarize.length > 0
				? summarizeHistory()
				: Promise.resolve({ text: 'No prior history.', usage: undefined }),
			summarize(
				'compaction_prefix',
				turnPrefixMessages,
				{ turnPrefix: true },
				prefixTokens,
				target,
				signal,
				observer,
				'Turn prefix summarization failed',
			),
		]);
		addCallUsage(historyResult.usage);
		addCallUsage(turnPrefixResult.usage);
		summary = `${historyResult.text}\n\n---\n\n**Turn Context (split turn):**\n\n${turnPrefixResult.text}`;
	} else {
		const historyResult = await summarizeHistory();
		addCallUsage(historyResult.usage);
		summary = historyResult.text;
	}

	const { readFiles, modifiedFiles } = computeFileLists(fileOps);
	summary += formatFileOperations(readFiles, modifiedFiles);

	return {
		summary,
		tokensBefore,
		details: { readFiles, modifiedFiles },
		usage: aggregateUsage,
	};
}
/**
 * Co/**
 * Context-overflow classification for an assistant message, with TanStack's
 * `isContextOverflow`. A structural check for the runtime's own Workers AI
 * binding marker comes first, so a binding 413 classifies without depending
 * on the pattern list, or on its non-overflow precedence (a 413 whose
 * provider body mentions "rate limit" must still classify as overflow).
 */
export function isAssistantContextOverflow(
	assistant: AssistantMessage,
	contextWindow: number,
): boolean {
	const isError = assistant.stopReason === 'error';
	if (isError && assistant.errorMessage?.includes(WORKERS_AI_OVERFLOW_MARKER)) {
		return true;
	}
	return isContextOverflow({
		...(isError ? { error: assistant.errorMessage } : {}),
		// pi's count for the silent checks: input and cache reads, no cache writes.
		usage: {
			promptTokens: assistant.usage.input + assistant.usage.cacheRead,
			completionTokens: assistant.usage.output,
		},
		finishReason: assistant.stopReason,
		contextWindow,
		provider: assistant.provider,
	});
}
