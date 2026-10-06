/**
 * The agent loop of a Flue session, run by TanStack's `HarnessSession`.
 *
 * The harness runs each turn: the model calls, the tool calls, and the
 * continuation when steered messages wait. Flue keeps the transcript
 * (`state.messages`), so each model call sends the context that Flue's
 * records hold. The listeners get pi's loop events in pi's order, and the
 * session builds its records and its observations from them.
 */
import {
	type AnyTextAdapter,
	type ChatMiddleware,
	EventType,
	type MidConversationChange,
	type ModelMessage,
	type StreamChunk,
} from '@tanstack/ai';
import { planMidConversationChanges } from '@tanstack/ai/adapter-internals';
import { createHarnessHost, defineHarness, type HarnessSession } from '@tanstack/ai-harness';
import type {
	AgentMessage,
	AgentTool,
	AgentToolResult,
	AssistantMessage,
	ImageContent,
	SystemMessage,
	ThinkingLevel,
	ToolCall,
	ToolDeclaration,
	ToolResultMessage,
} from './llm-types.ts';
import {
	type AssistantBlockEvent,
	AssistantStreamAssembler,
	modelInfo,
	toModelRequest,
} from './model-messages.ts';
import type { FlueModel } from './providers/provider.ts';
import { toHarnessTool } from './tool-bridge.ts';

/** A finished turn: the assistant message and the results of its tool calls. */
export interface CompletedTurn {
	message: AssistantMessage;
	toolResults: ToolResultMessage[];
}

/** The events of one run, in pi's order. */
export type AgentEvent =
	| { type: 'agent_start' }
	| { type: 'agent_end'; messages: AgentMessage[] }
	| { type: 'turn_start' }
	| { type: 'turn_end'; message: AgentMessage; toolResults: ToolResultMessage[] }
	| { type: 'message_start'; message: AgentMessage }
	| {
			type: 'message_update';
			message: AgentMessage;
			assistantMessageEvent: AssistantBlockEvent;
	  }
	| { type: 'message_end'; message: AgentMessage }
	| {
			type: 'tool_execution_start';
			toolCallId: string;
			toolName: string;
			args: unknown;
	  }
	| { type: 'tool_execution_update'; toolCallId: string; toolName: string }
	| {
			type: 'tool_execution_end';
			toolCallId: string;
			toolName: string;
			result: AgentToolResult;
			isError: boolean;
	  };

export type AgentEventListener = (event: AgentEvent, signal: AbortSignal) => Promise<void> | void;

/** What one model call sends, for the session's request observation. */
export interface ModelCallRequest {
	model: FlueModel;
	systemPrompt: string;
	messages: AgentMessage[];
	tools: AgentTool[];
	thinkingLevel: ThinkingLevel;
}

export interface AgentLoopOptions {
	initialState: {
		systemPrompt: string;
		model: FlueModel;
		tools: AgentTool[];
		messages?: AgentMessage[];
		thinkingLevel: ThinkingLevel;
	};
	/** The conversation's prompt cache key. */
	sessionId: string;
	/** The adapter for the model calls of one run of `model`. */
	createAdapter(model: FlueModel, signal: AbortSignal): Promise<AnyTextAdapter>;
	/** Runs before each model call. */
	onModelRequest?(request: ModelCallRequest): void;
	/** Runs before every turn of a run after the first, with the turn before it. */
	prepareNextTurn?(turn: CompletedTurn): Promise<unknown> | unknown;
}

/** pi's leading system message: the prompt and the tools the model may call. */
export function createInitialSystemMessage(
	systemPrompt: string | undefined,
	tools: readonly ToolDeclaration[],
): SystemMessage | undefined {
	if (!systemPrompt && tools.length === 0) return undefined;
	return {
		role: 'system',
		content: systemPrompt ?? '',
		...(tools.length > 0 ? { toolsAdded: [...tools] } : {}),
		timestamp: Date.now(),
	};
}

/** A tool as the model sees it. */
export function toToolDeclaration(tool: ToolDeclaration): ToolDeclaration {
	return {
		name: tool.name,
		description: tool.description,
		parameters: tool.parameters,
	};
}

function systemMessageText(message: SystemMessage) {
	return typeof message.content === 'string'
		? message.content
		: message.content.map((block) => block.text).join('');
}

const EMPTY_USAGE = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

function errorToolResult(message: string): AgentToolResult {
	return { content: [{ type: 'text', text: message }], details: {} };
}

function errorText(error: unknown) {
	return error instanceof Error ? error.message : String(error);
}

/** One tool batch: the calls of one assistant message. */
interface ToolBatch {
	assistant: AssistantMessage;
	calls: ToolCall[];
	/** pi's rule: the whole batch runs one call at a time when one called tool asks for it. */
	sequential: boolean;
	/** A `length` stop: the arguments may be cut, so no call runs. */
	truncated: boolean;
	finished: Map<string, { result: AgentToolResult; isError: boolean }>;
	/** Result messages that were already emitted (sequential batches emit each one at once). */
	emitted: Set<string>;
	/** Each call of a sequential batch waits for the one before it. */
	turns: PromiseWithResolvers<void>[];
}

/** The state of one run: one `prompt` or `continue`. */
interface LoopRun {
	signal: AbortSignal;
	adapter: AnyTextAdapter | undefined;
	/** The mid-conversation record of the model call in flight. */
	midConversationChange: MidConversationChange | undefined;
	/** Messages for the next model call: the steered ones and the prompt's queue. */
	pending: AgentMessage[];
	firstCall: boolean;
	lastTurn: CompletedTurn | undefined;
	assembler: AssistantStreamAssembler | undefined;
	assistantStarted: boolean;
	batch: ToolBatch | undefined;
	/** Every result of the last batch asked to stop. */
	terminated: boolean;
	/** An error or aborted assistant ended the run. */
	stopped: boolean;
	/** The first error that a listener threw. */
	failure: { error: unknown } | undefined;
	newMessages: AgentMessage[];
}

/**
 * A Flue session's loop on a TanStack `HarnessSession`. It has the surface of
 * pi's `Agent` that the session uses: `state`, `prompt`, `continue`,
 * `steer`, `waitForIdle`, `abort`, and `subscribe`.
 */
export class AgentLoop {
	readonly state: {
		readonly systemPrompt: string;
		model: FlueModel;
		thinkingLevel: ThinkingLevel;
		tools: AgentTool[];
		messages: AgentMessage[];
		errorMessage: string | undefined;
	};
	private readonly listeners = new Set<AgentEventListener>();
	private steering: AgentMessage[] = [];
	private active: { promise: Promise<void>; controller: AbortController } | undefined;
	private run: LoopRun | undefined;
	/** One event at a time, in emit order, as pi awaited each listener. */
	private emitting: Promise<void> = Promise.resolve();
	private harnessSession: Promise<HarnessSession> | undefined;
	/**
	 * The mid-conversation record of each assistant message: the tools and
	 * prompts its call declared. TanStack plans tool and prompt changes from
	 * these records. Kept in memory, as pi kept its transcript's tool
	 * declarations; a rebuilt context starts from the current set.
	 */
	private readonly midConversationChanges = new WeakMap<AgentMessage, MidConversationChange>();
	private readonly harnessTools = new WeakMap<AgentTool, ReturnType<typeof toHarnessTool>>();

	constructor(private readonly options: AgentLoopOptions) {
		let tools = options.initialState.tools.slice();
		let messages = options.initialState.messages?.slice() ?? [];
		const lead = createInitialSystemMessage(
			options.initialState.systemPrompt,
			tools.map(toToolDeclaration),
		);
		if (messages[0]?.role !== 'system' && lead) messages.unshift(lead);
		this.state = {
			get systemPrompt() {
				const lead = messages.find(
					(message): message is SystemMessage => message.role === 'system',
				);
				return lead ? systemMessageText(lead) : '';
			},
			model: options.initialState.model,
			thinkingLevel: options.initialState.thinkingLevel,
			get tools() {
				return tools;
			},
			set tools(next) {
				tools = next.slice();
			},
			get messages() {
				return messages;
			},
			set messages(next) {
				messages = next.slice();
			},
			errorMessage: undefined,
		};
	}

	subscribe(listener: AgentEventListener) {
		this.listeners.add(listener);
		return () => this.listeners.delete(listener);
	}

	/** Queue a message for the next model call of the running turn. */
	steer(message: AgentMessage) {
		this.steering.push(message);
	}

	clearSteeringQueue() {
		this.steering = [];
	}

	abort() {
		this.active?.controller.abort();
	}

	/** Resolves when the run and its listeners finished. */
	waitForIdle() {
		return this.active?.promise ?? Promise.resolve();
	}

	async prompt(input: string | AgentMessage[], images?: ImageContent[]) {
		if (this.active)
			throw new Error(
				'Agent is already processing a prompt. Use steer() to queue messages, or wait for completion.',
			);
		const messages: AgentMessage[] =
			typeof input === 'string'
				? [
						{
							role: 'user',
							content: [{ type: 'text', text: input }, ...(images ?? [])],
							timestamp: Date.now(),
						},
					]
				: input;
		await this.runLoop(messages, false);
	}

	/** Continue from the transcript. The last message must not be an assistant message, unless steered messages wait. */
	async continue() {
		if (this.active)
			throw new Error('Agent is already processing. Wait for completion before continuing.');
		const messages = this.state.messages;
		const last = messages.at(-1);
		if (!last || messages.every((message) => message.role === 'system'))
			throw new Error('No messages to continue from');
		if (last.role === 'assistant') {
			const queued = this.drainSteering();
			if (queued.length === 0) throw new Error('Cannot continue from message role: assistant');
			await this.runLoop(queued, true);
			return;
		}
		await this.runLoop([], false);
	}

	private drainSteering() {
		const drained = this.steering;
		this.steering = [];
		return drained;
	}

	private async runLoop(initial: AgentMessage[], skipInitialPoll: boolean) {
		const controller = new AbortController();
		const done = Promise.withResolvers<void>();
		this.active = { promise: done.promise, controller };
		this.state.errorMessage = undefined;
		const run: LoopRun = {
			signal: controller.signal,
			adapter: undefined,
			midConversationChange: undefined,
			pending: [],
			firstCall: true,
			lastTurn: undefined,
			assembler: undefined,
			assistantStarted: false,
			batch: undefined,
			terminated: false,
			stopped: false,
			failure: undefined,
			newMessages: [],
		};
		this.run = run;
		try {
			await this.emit({ type: 'agent_start' });
			await this.emit({ type: 'turn_start' });
			for (const message of initial) await this.emitMessage(message);
			run.pending = skipInitialPoll ? [] : this.drainSteering();
			await this.runTurn(run);
			if (run.failure) throw run.failure.error;
			await this.emit({ type: 'agent_end', messages: run.newMessages });
		} catch (error) {
			await this.failRun(error, controller.signal.aborted);
		} finally {
			this.run = undefined;
			this.active = undefined;
			done.resolve();
		}
	}

	/** One harness turn. The middleware below turns its hooks into the loop's events. */
	private async runTurn(run: LoopRun) {
		const session = await this.session();
		const adapter = await this.options.createAdapter(this.state.model, run.signal);
		run.adapter = adapter;
		const operation = session.prompt(
			// The model context comes from `state.messages` (see `onConfig`); this text only opens the turn.
			'Continue.',
			{
				overrides: {
					adapter,
					promptCache: { key: this.options.sessionId },
				},
			},
		);
		const cancel = () => void operation.cancel();
		run.signal.addEventListener('abort', cancel, { once: true });
		try {
			await operation;
		} catch {
			// The run's events already tell how it ended; see `finishAfterCancel`.
		} finally {
			run.signal.removeEventListener('abort', cancel);
		}
		if (run.signal.aborted) await this.finishAfterCancel(run);
	}

	private session() {
		this.harnessSession ??= createHarnessHost().open(
			defineHarness({
				name: 'flue/session',
				middleware: [this.middleware()],
				// pi's loop had no iteration limit.
				agentLoopStrategy: () => true,
				turn: {
					beforeFinish: () => this.beforeFinish(),
					maxFinishCycles: Number.MAX_SAFE_INTEGER,
				},
			}),
			{ threadId: this.options.sessionId },
		);
		return this.harnessSession;
	}

	private middleware(): ChatMiddleware {
		return {
			name: 'flue/agent-loop',
			onConfig: async (ctx) => {
				const run = this.run;
				if (!run || ctx.phase !== 'beforeModel') return;
				return this.guard(run, () => this.beforeModelCall(run));
			},
			onChunk: async (_ctx, chunk) => {
				const run = this.run;
				if (!run?.assembler) return;
				await this.guard(run, () => this.modelChunk(run, chunk));
			},
			onToolPhaseComplete: async (_ctx, info) => {
				const run = this.run;
				if (!run?.batch) return;
				const chatErrors = new Map(info.results.map((entry) => [entry.toolCallId, entry.result]));
				await this.guard(run, () => this.finishBatch(run, chatErrors));
			},
			onShouldContinue: () => {
				const run = this.run;
				return !run || (!run.stopped && !run.terminated && !run.failure);
			},
		};
	}

	/** Run listener work; the first error stops the run, as a throw in pi's loop did. */
	private async guard<T>(run: LoopRun, work: () => Promise<T>) {
		if (run.failure) throw run.failure.error;
		try {
			return await work();
		} catch (error) {
			run.failure ??= { error };
			throw error;
		}
	}

	private async beforeModelCall(run: LoopRun) {
		if (!run.firstCall) {
			if (run.lastTurn) await this.options.prepareNextTurn?.(run.lastTurn);
			if (run.pending.length === 0) run.pending = this.drainSteering();
			await this.emit({ type: 'turn_start' });
		}
		run.firstCall = false;
		run.terminated = false;
		const pending = run.pending;
		run.pending = [];
		for (const message of pending) await this.emitMessage(message);

		const { model, thinkingLevel } = this.state;
		const tools = this.state.tools.slice();
		const info = modelInfo(model);
		this.options.onModelRequest?.({
			model,
			systemPrompt: this.state.systemPrompt,
			messages: this.state.messages.slice(),
			tools,
			thinkingLevel,
		});
		const request = toModelRequest(
			{
				systemPrompt: this.state.systemPrompt,
				messages: this.state.messages,
				tools: tools.map(toToolDeclaration),
			},
			info,
		);
		const providerMessages = this.withMidConversationChanges(request.messages);
		const channels = run.adapter?.midConversationChannels;
		run.midConversationChange =
			channels?.tools || channels?.systemPrompts
				? planMidConversationChanges({
						messages: providerMessages,
						toolNames: tools.map((tool) => tool.name),
						systemPrompts: request.systemPrompts,
					}).record
				: undefined;
		run.assembler = new AssistantStreamAssembler(info);
		run.assistantStarted = false;
		return {
			providerMessages,
			systemPrompts: request.systemPrompts,
			tools: tools.map((tool) => this.harnessTool(tool)),
			// pi sent the off value for `off` too (for example Anthropic `thinking.type: 'disabled'`).
			reasoning: { level: thinkingLevel, summary: true },
		};
	}

	private async modelChunk(run: LoopRun, chunk: StreamChunk) {
		const assembler = run.assembler;
		if (!assembler) return;
		const events = assembler.push(chunk);
		if (!run.assistantStarted) {
			run.assistantStarted = true;
			const partial = events[0]?.partial;
			await this.emit({
				type: 'message_start',
				message: partial ? { ...partial } : assembler.snapshot(),
			});
		}
		for (const event of events)
			await this.emit({
				type: 'message_update',
				assistantMessageEvent: event,
				message: { ...event.partial },
			});
		const isEnd = chunk.type === EventType.RUN_FINISHED || chunk.type === EventType.RUN_ERROR;
		if (!isEnd) return;
		run.assembler = undefined;
		await this.finishAssistant(run, assembler.finish());
	}

	/** The request messages with each assistant's mid-conversation record, as `chat()` keeps them. */
	private withMidConversationChanges(messages: ModelMessage[]) {
		const assistants = this.state.messages.filter((message) => message.role === 'assistant');
		let index = 0;
		return messages.map((message) => {
			if (message.role !== 'assistant') return message;
			const source = assistants[index++];
			const change = source && this.midConversationChanges.get(source);
			return change ? { ...message, midConversationChange: change } : message;
		});
	}

	private async finishAssistant(run: LoopRun, message: AssistantMessage) {
		if (run.midConversationChange)
			this.midConversationChanges.set(message, run.midConversationChange);
		run.midConversationChange = undefined;
		await this.emit({ type: 'message_end', message });
		run.newMessages.push(message);
		if (message.stopReason === 'error' || message.stopReason === 'aborted') {
			run.stopped = true;
			run.lastTurn = { message, toolResults: [] };
			await this.emit({ type: 'turn_end', message, toolResults: [] });
			return;
		}
		const calls = message.content.filter((block): block is ToolCall => block.type === 'toolCall');
		if (calls.length === 0) {
			run.lastTurn = { message, toolResults: [] };
			await this.emit({ type: 'turn_end', message, toolResults: [] });
			run.pending = this.drainSteering();
			return;
		}
		const tools = this.state.tools;
		run.batch = {
			assistant: message,
			calls,
			sequential: calls.some(
				(call) => tools.find((tool) => tool.name === call.name)?.executionMode === 'sequential',
			),
			truncated: message.stopReason === 'length',
			finished: new Map(),
			emitted: new Set(),
			turns: calls.map(() => Promise.withResolvers<void>()),
		};
	}

	/** The harness tool for `tool`: it runs Flue's tool with pi's batch rules and events. */
	private harnessTool(tool: AgentTool) {
		let harnessTool = this.harnessTools.get(tool);
		if (harnessTool) return harnessTool;
		const loop = this;
		harnessTool = toHarnessTool({
			...tool,
			async execute(toolCallId, args, signal) {
				const run = loop.run;
				if (!run?.batch) return tool.execute(toolCallId, args, signal);
				return loop.executeCall(run, run.batch, tool, toolCallId, args, signal);
			},
		});
		this.harnessTools.set(tool, harnessTool);
		return harnessTool;
	}

	private async executeCall(
		run: LoopRun,
		batch: ToolBatch,
		tool: AgentTool,
		toolCallId: string,
		args: unknown,
		signal: AbortSignal | undefined,
	) {
		const index = batch.calls.findIndex((call) => call.id === toolCallId);
		if (batch.sequential && index > 0) await batch.turns[index - 1]?.promise;
		try {
			// pi stopped a sequential batch at an abort: later calls never start.
			if (batch.sequential && run.signal.aborted) return errorToolResult('Operation aborted');
			await this.guard(run, () =>
				this.emit({ type: 'tool_execution_start', toolCallId, toolName: tool.name, args }),
			);
			const outcome = batch.truncated
				? {
						result: errorToolResult(
							`Tool call "${tool.name}" was not executed: the response hit the output token limit, so its arguments may be truncated. Re-issue the tool call with complete arguments.`,
						),
						isError: true,
					}
				: await runTool(tool, toolCallId, args, signal);
			await this.guard(run, () => this.finishCall(batch, toolCallId, tool.name, outcome));
			return outcome.result;
		} finally {
			batch.turns[index]?.resolve();
		}
	}

	private async finishCall(
		batch: ToolBatch,
		toolCallId: string,
		toolName: string,
		outcome: { result: AgentToolResult; isError: boolean },
	) {
		batch.finished.set(toolCallId, outcome);
		await this.emit({
			type: 'tool_execution_end',
			toolCallId,
			toolName,
			result: outcome.result,
			isError: outcome.isError,
		});
		if (batch.sequential || batch.truncated)
			await this.emitResultMessage(batch, toolCallId, toolName, outcome);
	}

	private async emitResultMessage(
		batch: ToolBatch,
		toolCallId: string,
		toolName: string,
		outcome: { result: AgentToolResult; isError: boolean },
	) {
		batch.emitted.add(toolCallId);
		const message: ToolResultMessage = {
			role: 'toolResult',
			toolCallId,
			toolName,
			content: outcome.result.content ?? [],
			details: outcome.result.details,
			...(outcome.result.usage ? { usage: outcome.result.usage } : {}),
			isError: outcome.isError,
			timestamp: Date.now(),
		};
		await this.emitMessage(message);
		return message;
	}

	/**
	 * The end of a tool batch: calls that never ran get pi's error result
	 * (an unknown tool, or arguments that failed the check), the result
	 * messages of a parallel batch go out in call order, and the turn ends.
	 */
	private async finishBatch(run: LoopRun, chatResults: Map<string, unknown>) {
		const batch = run.batch;
		if (!batch) return;
		run.batch = undefined;
		const toolResults: ToolResultMessage[] = [];
		const aborted = run.signal.aborted;
		for (const call of batch.calls) {
			let outcome = batch.finished.get(call.id);
			if (!outcome) {
				// pi never ran the calls after an abort, and recorded nothing for them.
				if (aborted) continue;
				await this.emit({
					type: 'tool_execution_start',
					toolCallId: call.id,
					toolName: call.name,
					args: call.arguments,
				});
				outcome = {
					result: errorToolResult(skippedCallText(call, chatResults.get(call.id))),
					isError: true,
				};
				await this.finishCall(batch, call.id, call.name, outcome);
			}
			if (!batch.emitted.has(call.id))
				toolResults.push(await this.emitResultMessage(batch, call.id, call.name, outcome));
			else toolResults.push(this.resultMessageOf(call, outcome));
		}
		run.terminated =
			toolResults.length > 0 &&
			[...batch.finished.values()].every(({ result }) => result.terminate === true);
		run.lastTurn = { message: batch.assistant, toolResults };
		await this.emit({ type: 'turn_end', message: batch.assistant, toolResults });
		run.pending = this.drainSteering();
	}

	/** The result message a sequential batch already emitted for `call`. */
	private resultMessageOf(call: ToolCall, outcome: { result: AgentToolResult; isError: boolean }) {
		const message = this.state.messages.findLast(
			(entry): entry is ToolResultMessage =>
				entry.role === 'toolResult' && entry.toolCallId === call.id,
		);
		return (
			message ?? {
				role: 'toolResult' as const,
				toolCallId: call.id,
				toolName: call.name,
				content: outcome.result.content,
				details: outcome.result.details,
				isError: outcome.isError,
				timestamp: Date.now(),
			}
		);
	}

	/** The model stopped calling tools: continue the same run while steered messages wait. */
	private beforeFinish() {
		const run = this.run;
		if (!run || run.stopped || run.failure || run.signal.aborted) return undefined;
		// pi polled the steering queue once, after `turn_end`.
		if (run.pending.length === 0) return undefined;
		return { messages: [{ role: 'user' as const, content: 'Continue.' }] };
	}

	/**
	 * After a cancel: close what the harness cut. A model call that was
	 * streaming ends as an aborted assistant, and a tool batch ends with the
	 * results it has, as pi's loop did.
	 */
	private async finishAfterCancel(run: LoopRun) {
		if (run.failure) return;
		try {
			if (run.batch) await this.finishBatch(run, new Map());
			if (run.failure || run.stopped) return;
			const assembler = run.assembler;
			run.assembler = undefined;
			if (assembler) {
				if (!run.assistantStarted)
					await this.emit({ type: 'message_start', message: assembler.snapshot() });
				assembler.push({
					type: EventType.RUN_ERROR,
					timestamp: Date.now(),
					message: 'Request was aborted.',
					code: 'aborted',
				});
				await this.finishAssistant(run, assembler.finish());
				return;
			}
			// pi's next model call saw the abort and ended as an aborted assistant.
			if (run.lastTurn) await this.options.prepareNextTurn?.(run.lastTurn);
			await this.emit({ type: 'turn_start' });
			const aborted = this.failureMessage('Request was aborted.', true);
			await this.emit({ type: 'message_start', message: aborted });
			await this.finishAssistant(run, aborted);
		} catch (error) {
			run.failure ??= { error };
		}
	}

	private failureMessage(errorMessage: string, aborted: boolean): AssistantMessage {
		const { model } = this.state;
		return {
			role: 'assistant',
			content: [{ type: 'text', text: '' }],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: EMPTY_USAGE,
			stopReason: aborted ? 'aborted' : 'error',
			errorMessage,
			timestamp: Date.now(),
		};
	}

	/** pi's run failure: one failure assistant, then the end of the turn and the run. */
	private async failRun(error: unknown, aborted: boolean) {
		const message = this.failureMessage(errorText(error), aborted);
		try {
			await this.emit({ type: 'message_start', message });
			await this.emit({ type: 'message_end', message });
			await this.emit({ type: 'turn_end', message, toolResults: [] });
			await this.emit({ type: 'agent_end', messages: [message] });
		} catch {
			// pi ignored listener errors on the failure path too.
		}
	}

	private async emitMessage(message: AgentMessage) {
		await this.emit({ type: 'message_start', message });
		await this.emit({ type: 'message_end', message });
		this.run?.newMessages.push(message);
	}

	/** Apply the event to the state, then await each listener, one event at a time. */
	private emit(event: AgentEvent) {
		const next = this.emitting.then(() => this.deliver(event));
		// A failed listener fails this event, not the ones after it.
		this.emitting = next.catch(() => undefined);
		return next;
	}

	private async deliver(event: AgentEvent) {
		if (event.type === 'message_end') this.state.messages.push(event.message);
		if (
			event.type === 'turn_end' &&
			event.message.role === 'assistant' &&
			event.message.errorMessage
		)
			this.state.errorMessage = event.message.errorMessage;
		const signal = this.active?.controller.signal;
		if (!signal) throw new Error('Agent listener invoked outside active run');
		for (const listener of this.listeners) await listener(event, signal);
	}
}

async function runTool(
	tool: AgentTool,
	toolCallId: string,
	args: unknown,
	signal: AbortSignal | undefined,
) {
	try {
		const prepared = tool.prepareArguments ? tool.prepareArguments(args) : args;
		const result = await tool.execute(toolCallId, prepared, signal, () => {});
		return { result, isError: false };
	} catch (error) {
		return { result: errorToolResult(errorText(error)), isError: true };
	}
}

/** The text of a call that the harness never ran: an unknown tool, or arguments it refused. */
function skippedCallText(call: ToolCall, chatResult: unknown) {
	if (
		typeof chatResult === 'object' &&
		chatResult !== null &&
		'error' in chatResult &&
		typeof chatResult.error === 'string'
	)
		return chatResult.error;
	return `Tool ${call.name} not found`;
}
