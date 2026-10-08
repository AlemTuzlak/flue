/**
 * The agent loop of a Flue session, run by TanStack's `HarnessSession`.
 *
 * The harness runs each turn: the model calls, the tool calls, and the
 * continuation when steered messages wait. The listeners get pi's loop
 * events in pi's order, and the session builds its records and its
 * observations from them.
 *
 * With a durable binding, the loop runs on a thread of the instance host,
 * and the harness transcript is the model context. Without one, the loop
 * runs on an in-memory host, and each model call sends the context that
 * Flue's own transcript (`state.messages`) holds.
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
import {
	createHarnessHost,
	defineHarness,
	type FinishContext,
	type HarnessSession,
	type HarnessTurnOptions,
	type JoinContext,
	type ModelErrorContext,
	type RecoverContext,
	type RecoverHook,
	type TurnAdditions,
	type UserInput,
} from '@tanstack/ai-harness';
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
import { renderSignalMessage } from './message-rendering.ts';
import {
	type AssistantBlockEvent,
	AssistantStreamAssembler,
	fromModelMessage,
	modelInfo,
	toModelContext,
	toModelRequest,
	toUserInput,
} from './model-messages.ts';
import type { FlueModel } from './providers/provider.ts';
import type { createInstanceHarnessHost } from './runtime/instance-harness-host.ts';
import { isRetryableModelError } from './submission-state.ts';
import { type HarnessToolOptions, toHarnessTool } from './tool-bridge.ts';

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

/** The thread of a loop on the durable host of its agent instance. */
export interface AgentLoopDurableBinding {
	/** The host from `createInstanceHarnessHost`. */
	host: ReturnType<typeof createInstanceHarnessHost>;
	/** The conversation id. */
	threadId: string;
	/** The instance stream path: the log of the host. */
	logId: string;
}

/** Options of one harness input of the loop. */
export interface AgentLoopInputOptions {
	/** The input id. The loop makes one when it is not set. */
	inputId?: string;
	/** Messages for the model calls of this input only. No store keeps them. */
	ephemeral?: readonly ModelMessage[];
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
	/**
	 * The durable thread of the loop. Without it, the loop runs on an
	 * in-memory host, and the model context comes from `state.messages`.
	 */
	durable?: AgentLoopDurableBinding;
	/**
	 * How an input that a crashed host left recovers. See `durability.recover`.
	 * On a durable binding, a turn that runs gets the loop's adapter and
	 * tools, and {@link AgentLoop.recoverTurns} follows it.
	 */
	recover?: RecoverHook;
	/**
	 * On a durable binding: runs once the loop opened its thread, before any
	 * input of the loop reaches the harness. A run waits for it. `inputs` are
	 * the messages of `state.messages` that the run sends to the harness as
	 * its inputs.
	 */
	onOpen?(session: HarnessSession, inputs: readonly AgentMessage[]): Promise<void>;
	/** Whether a waiting input joins the running turn now. */
	canJoin?: HarnessTurnOptions['canJoin'];
	/**
	 * Runs before the model call that joined inputs reach. Its records land
	 * in the append that joins them.
	 */
	onJoin?: HarnessTurnOptions['onJoin'];
	/**
	 * On a durable binding: runs before the model call that joined inputs
	 * reach, after the append that joined them, with their input ids.
	 */
	onJoined?(inputIds: readonly string[]): Promise<void>;
	/**
	 * On a durable binding: the model stopped calling tools. Return records
	 * or ephemeral messages to send the model back to work in the same turn.
	 * An ephemeral message reaches every later model call of the run, at the
	 * place where it first went out.
	 */
	beforeFinish?(ctx: FinishContext): Promise<TurnAdditions | undefined>;
	/**
	 * On a durable binding: runs before each model call with the harness
	 * transcript. A list it returns replaces the transcript (a compaction).
	 */
	rewriteContext?(messages: readonly ModelMessage[]): Promise<ModelMessage[] | undefined>;
	/**
	 * What a failed model call does. The default on a durable binding is
	 * {@link retryModelErrors}. Without a binding, a failed call ends the run.
	 */
	onModelError?: HarnessTurnOptions['onModelError'];
	/** Runs before each bridged tool call, with its step and its staged records. */
	onToolCall?: HarnessToolOptions['onCall'];
	/** Gets the whole result of each bridged tool call. */
	onToolResult?: HarnessToolOptions['onResult'];
}

/** Flue's retries of one failed model call, after the last finished tool phase. */
const MAX_MODEL_RETRIES = 3;
const RETRY_BASE_DELAY_MS = 2_000;

/**
 * Flue's `turn.onModelError`: retry a model error that
 * `isRetryableModelError` accepts, at most 3 times after the last finished
 * tool phase. The backoff is `2000 ms * 2^(n-1)` for retry `n`, times a
 * jitter from 0.75 to 1.0. The text of the failed call is dropped.
 */
export async function retryModelErrors(ctx: ModelErrorContext) {
	if (ctx.retries >= MAX_MODEL_RETRIES || !isRetryableModelError(ctx.error)) return undefined;
	const jitter = 0.75 + Math.random() * 0.25;
	await abortableDelay(Math.round(RETRY_BASE_DELAY_MS * 2 ** ctx.retries * jitter), ctx.signal);
	if (ctx.signal.aborted) return undefined;
	return 'retry' as const;
}

function abortableDelay(ms: number, signal: AbortSignal) {
	return new Promise<void>((resolve) => {
		if (signal.aborted) return resolve();
		const done = () => {
			clearTimeout(timer);
			signal.removeEventListener('abort', done);
			resolve();
		};
		const timer = setTimeout(done, ms);
		signal.addEventListener('abort', done, { once: true });
	});
}

/** The result text of a tool call of an answer that hit the output token limit. */
function truncatedCallText(toolName: string) {
	return `Tool call "${toolName}" was not executed: the response hit the output token limit, so its arguments may be truncated. Re-issue the tool call with complete arguments.`;
}

/** The result of a cut tool call that must not run twice: Flue's interrupted marker. */
export const INTERRUPTED_TOOL_RESULT = JSON.stringify({
	type: 'interrupted',
	message: 'Tool execution was interrupted before completion. The outcome is unknown.',
});

/**
 * Flue's two signals after a cut answer: the history records of a continued
 * answer, and the notes the model gets after the cut text.
 */
export const STREAM_RECOVERY_SIGNALS = [
	{ type: 'stream_interrupted', content: 'The previous assistant stream was interrupted.' },
	{ type: 'stream_continued', content: 'Continue from the durable partial assistant response.' },
] as const;

/** The signals of {@link STREAM_RECOVERY_SIGNALS} as the model reads them. */
const STREAM_RECOVERY_NOTES = STREAM_RECOVERY_SIGNALS.map(({ type, content }) =>
	renderSignalMessage({ role: 'signal', type, content, timestamp: 0 }),
);

function createInputId() {
	return `flue:${crypto.randomUUID()}`;
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
	/** The calls of a cut batch that the harness closed with an error result. */
	closed: Set<string>;
	finished: Map<string, { result: AgentToolResult; isError: boolean }>;
	/** Result messages that were already emitted (sequential batches emit each one at once). */
	emitted: Set<string>;
	/** Each call of a sequential batch waits for the one before it. */
	turns: PromiseWithResolvers<void>[];
}

/** The harness input that starts a run. `recover` follows the turns that recovery runs. */
type RunInput =
	| { kind: 'prompt'; message: UserInput; inputId: string; ephemeral?: readonly ModelMessage[] }
	| { kind: 'continue'; inputId: string; ephemeral?: readonly ModelMessage[] }
	| { kind: 'recover' };

/** The outcome of one tool call. */
export interface ToolCallOutcome {
	result: AgentToolResult;
	isError: boolean;
}

/** An ephemeral message of a `beforeFinish` hook, and where it went out. */
interface Reminder {
	messages: readonly ModelMessage[];
	/** The transcript length at the model call it first reached. Unset until then. */
	at: number | undefined;
}

/** A message waiting for the next run, with its harness input id. */
interface QueuedSteer {
	message: AgentMessage;
	inputId: string | undefined;
}

/** The state of one run: one `prompt` or `continue`. */
interface LoopRun {
	signal: AbortSignal;
	/** Resolves when the run's input reached the harness, so a steer joins it. */
	started: PromiseWithResolvers<void>;
	/** Steers sent to the harness that it has not stored yet. */
	admissions: Promise<unknown>[];
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
	/** Input ids that joined since the last model call. */
	joined: string[];
	/** The ephemeral messages of `beforeFinish`, for the later model calls of the run. */
	reminders: Reminder[];
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
	private steering: QueuedSteer[] = [];
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
	/** Steered messages sent to the harness, by input id, until they join a model call. */
	private readonly joining = new Map<string, AgentMessage>();
	/** The ids of the inputs that recovery runs, until a run follows them. */
	private recovered: string[] = [];

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

	/**
	 * Queue a message for the next model call of the running turn. On a
	 * durable binding, a steer during a run goes to the harness as an input
	 * with `busy: 'steer'`, and joins the next model call.
	 */
	steer(message: AgentMessage, options: { inputId?: string } = {}) {
		const run = this.run;
		if (this.options.durable && run) {
			this.sendSteer(run, message, options.inputId ?? createInputId());
			return;
		}
		this.steering.push({ message, inputId: options.inputId });
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

	/**
	 * Run a turn for `input`. The harness gets the real user message, with
	 * the images as content parts, as an input with `busy: 'steer'`.
	 */
	async prompt(
		input: string | AgentMessage[],
		images?: ImageContent[],
		options: AgentLoopInputOptions = {},
	) {
		this.assertIdle(
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
		const [first, ...rest] = messages;
		if (!first) throw new Error('No messages to prompt with');
		const durable = this.options.durable;
		// The harness takes one message per input. On a binding, the others join as steers.
		if (durable) this.steering.unshift(...rest.map((message) => ({ message, inputId: undefined })));
		// Without a binding, the context comes from `state.messages`, so any message can open the turn.
		const isInput = durable || first.role === 'user' || first.role === 'signal';
		await this.runLoop(durable ? [first] : messages, false, {
			kind: 'prompt',
			message: isInput ? toUserInput(first) : 'Continue.',
			inputId: options.inputId ?? createInputId(),
			...(options.ephemeral ? { ephemeral: options.ephemeral } : {}),
		});
	}

	/** Continue from the transcript. The last message must not be an assistant message, unless steered messages wait. */
	async continue() {
		this.assertIdle('Agent is already processing. Wait for completion before continuing.');
		const messages = this.state.messages;
		const last = messages.at(-1);
		if (!last || messages.every((message) => message.role === 'system'))
			throw new Error('No messages to continue from');
		if (last.role === 'assistant') {
			const queued = this.drainSteering();
			const [first, ...rest] = queued;
			if (!first) throw new Error('Cannot continue from message role: assistant');
			if (this.options.durable) {
				this.steering.unshift(...rest);
				await this.runLoop([first.message], false, {
					kind: 'prompt',
					message: toUserInput(first.message),
					inputId: first.inputId ?? createInputId(),
				});
				return;
			}
			await this.runLoop(
				queued.map((steer) => steer.message),
				true,
				this.legacyContinueInput(),
			);
			return;
		}
		if (this.options.durable) {
			await this.continueTurn();
			return;
		}
		await this.runLoop([], false, this.legacyContinueInput());
	}

	/**
	 * Run a turn from the harness transcript, with no new message. The
	 * harness refuses it unless the transcript ends with a user or a tool
	 * message.
	 */
	async continueTurn(options: AgentLoopInputOptions = {}) {
		this.assertIdle('Agent is already processing. Wait for completion before continuing.');
		await this.runLoop([], false, {
			kind: 'continue',
			inputId: options.inputId ?? createInputId(),
			...(options.ephemeral ? { ephemeral: options.ephemeral } : {}),
		});
	}

	/**
	 * On a durable binding: open the thread, and follow the turns that
	 * recovery runs there (the inputs a stopped host left), with the loop's
	 * events. Resolves when they ended. With no such turn, the run ends with
	 * no model call.
	 */
	async recoverTurns() {
		this.assertIdle('Agent is already processing. Wait for completion before recovering.');
		await this.runLoop([], false, { kind: 'recover' });
	}

	/**
	 * How the harness input `inputId` of the loop's thread ended. Waits until
	 * it ends; see `HarnessSession.settled`.
	 */
	async settled(inputId: string) {
		const session = await this.session();
		return session.settled(inputId);
	}

	/**
	 * In the `recover` hook of a durable binding: the recovered turn runs the
	 * calls of `assistant` again that have no outcome in `outcomes`. The batch
	 * then ends with the loop's events, as a live batch does. Each call in
	 * `outcomes` keeps its outcome and does not run.
	 */
	resumeToolBatch(assistant: AssistantMessage, outcomes: ReadonlyMap<string, ToolCallOutcome>) {
		const run = this.run;
		if (!run) throw new Error('[flue] A tool batch can resume only in a running recovery.');
		const batch = this.createBatch(assistant);
		for (const [toolCallId, outcome] of outcomes) {
			batch.finished.set(toolCallId, outcome);
			batch.emitted.add(toolCallId);
		}
		run.batch = batch;
		// The model call after the batch starts a new turn.
		run.firstCall = false;
	}

	private assertIdle(message: string) {
		if (this.active) throw new Error(message);
	}

	/**
	 * A continue without a binding. Its context comes from `state.messages`,
	 * which the harness transcript does not hold, so it opens the turn with a
	 * prompt, as the harness transcript can end with an answer.
	 */
	private legacyContinueInput(): RunInput {
		return { kind: 'prompt', message: 'Continue.', inputId: createInputId() };
	}

	private drainSteering() {
		const drained = this.steering;
		this.steering = [];
		return drained;
	}

	/** Send a steered message to the harness. It joins the running turn. */
	private sendSteer(run: LoopRun, message: AgentMessage, inputId: string) {
		this.joining.set(inputId, message);
		const input = toUserInput(message);
		const admission = run.started.promise.then(async () => {
			const session = await this.session();
			const operation = session.prompt(input, { inputId, busy: 'steer' });
			// The steer settles with the turn it joins; the run reports that turn.
			operation.then(undefined, () => undefined);
			await operation.receipt;
		});
		run.admissions.push(admission.catch(() => undefined));
	}

	/** Wait until the harness stored the steers sent so far, so the turn can join them. */
	private async flushAdmissions(run: LoopRun) {
		while (run.admissions.length > 0) {
			const admissions = run.admissions;
			run.admissions = [];
			await Promise.all(admissions);
		}
	}

	private async runLoop(initial: AgentMessage[], skipInitialPoll: boolean, input: RunInput) {
		const controller = new AbortController();
		const done = Promise.withResolvers<void>();
		this.active = { promise: done.promise, controller };
		this.state.errorMessage = undefined;
		const run: LoopRun = {
			signal: controller.signal,
			started: Promise.withResolvers<void>(),
			admissions: [],
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
			joined: [],
			reminders: [],
		};
		this.run = run;
		try {
			await this.emit({ type: 'agent_start' });
			await this.emit({ type: 'turn_start' });
			for (const message of initial) await this.emitMessage(message);
			const queued = skipInitialPoll ? [] : this.drainSteering();
			if (this.options.durable)
				for (const steer of queued)
					this.sendSteer(run, steer.message, steer.inputId ?? createInputId());
			else run.pending = queued.map((steer) => steer.message);
			await this.runTurn(run, input);
			if (run.failure) throw run.failure.error;
			await this.emit({ type: 'agent_end', messages: run.newMessages });
		} catch (error) {
			await this.failRun(error, controller.signal.aborted);
		} finally {
			run.started.resolve();
			// A steer that joined no model call of this run runs as its own harness turn.
			this.joining.clear();
			this.run = undefined;
			this.active = undefined;
			done.resolve();
		}
	}

	/** One harness turn. The middleware below turns its hooks into the loop's events. */
	private async runTurn(run: LoopRun, input: RunInput) {
		const session = await this.session();
		if (input.kind === 'recover') return this.followRecoveredTurns(run, session);
		const adapter = await this.options.createAdapter(this.state.model, run.signal);
		run.adapter = adapter;
		const overrides = {
			adapter,
			promptCache: { key: this.options.sessionId },
		};
		const ephemeral = input.ephemeral ? { ephemeral: input.ephemeral } : {};
		const operation =
			input.kind === 'prompt'
				? session.prompt(input.message, {
						inputId: input.inputId,
						busy: 'steer',
						overrides,
						...ephemeral,
					})
				: session.continue({ inputId: input.inputId, overrides, ...ephemeral });
		run.started.resolve();
		const cancel = () => void operation.cancel();
		run.signal.addEventListener('abort', cancel, { once: true });
		let failure: { error: unknown } | undefined;
		try {
			await operation;
		} catch (error) {
			// The run's events already tell how it ended; see `finishAfterCancel`.
			failure = { error };
		} finally {
			run.signal.removeEventListener('abort', cancel);
		}
		if (run.signal.aborted) {
			await this.finishAfterCancel(run);
			return;
		}
		// The harness session closed under the run (its host stopped): the run
		// fails, and a later run opens the thread again.
		if (failure && operation.status() === 'cancelled') {
			this.harnessSession = undefined;
			run.failure ??= failure;
			return;
		}
		// A cut batch that no later model call closed: the harness ended the turn.
		if (run.batch && !run.failure) await this.guard(run, () => this.finishBatch(run, new Map()));
		// The harness refused the input before any model call (for example `nothing_to_continue`).
		if (failure && run.firstCall) run.failure ??= failure;
	}

	/** Wait for the turns that recovery runs. Their events reach `run` through the middleware. */
	private async followRecoveredTurns(run: LoopRun, session: HarnessSession) {
		run.started.resolve();
		const inputIds = this.recovered;
		this.recovered = [];
		const cancel = () => void session.cancel();
		run.signal.addEventListener('abort', cancel, { once: true });
		let failure: { error: unknown } | undefined;
		try {
			for (const inputId of inputIds) {
				const settlement = await session.settled(inputId);
				if (settlement.outcome === 'failed')
					failure ??= {
						error: new Error(settlement.error?.message ?? 'The recovered turn failed.'),
					};
			}
		} catch (error) {
			failure ??= { error };
		} finally {
			run.signal.removeEventListener('abort', cancel);
		}
		if (run.signal.aborted) {
			await this.finishAfterCancel(run);
			return;
		}
		if (run.batch && !run.failure) await this.guard(run, () => this.finishBatch(run, new Map()));
		// A failure that no model call of this run reported.
		if (failure && run.firstCall) run.failure ??= failure;
	}

	/**
	 * The harness `recover` on a durable binding. The `recover` option
	 * decides first. A turn that runs gets the loop's adapter, and the loop
	 * follows it (see {@link recoverTurns}).
	 */
	private async recoverInput(ctx: RecoverContext) {
		const decision = (await this.options.recover?.(ctx)) ?? ctx.decision;
		if (decision.action !== 'run') return decision;
		this.recovered.push(ctx.input.inputId);
		const signal = this.run?.signal ?? new AbortController().signal;
		const adapter = await this.options.createAdapter(this.state.model, signal);
		return {
			action: 'run' as const,
			overrides: {
				adapter,
				promptCache: { key: this.options.sessionId },
				// A cut call runs again before the first model call sets the tools.
				tools: this.state.tools.map((tool) => this.harnessTool(tool)),
				...decision.overrides,
			},
		};
	}

	private session() {
		if (this.harnessSession) return this.harnessSession;
		const opening = this.openSession();
		this.harnessSession = opening;
		// A failed open does not stay: the next run opens the thread again.
		opening.catch(() => {
			if (this.harnessSession === opening) this.harnessSession = undefined;
		});
		return opening;
	}

	/** Open the loop's thread, and run the `onOpen` option on a durable binding. */
	private async openSession() {
		const session = await this.openThread();
		if (this.options.durable) await this.options.onOpen?.(session, this.run?.newMessages ?? []);
		return session;
	}

	/** The loop's thread: on the durable host of the binding, or on an in-memory host. */
	private openThread() {
		const durable = this.options.durable;
		const host = durable?.host ?? createHarnessHost();
		const { canJoin, onModelError } = this.options;
		const modelErrors = onModelError ?? (durable ? retryModelErrors : undefined);
		const harness = defineHarness({
			name: 'flue/session',
			middleware: [this.middleware()],
			// pi's loop had no iteration limit.
			agentLoopStrategy: () => true,
			durability: {
				// On a durable binding, Flue's submission ledger decides a
				// recovered input in the `recover` hook, by its own attempt
				// budget and deadline. The harness limits never decide first:
				// no attempt limit, and no time limit (which would also abort
				// a live turn).
				...(durable
					? { maxAttempts: Number.MAX_SAFE_INTEGER }
					: { maxAttempts: 10, timeoutMs: 3_600_000 }),
				// A cut answer keeps its calls; each gets this error result, and the model goes on.
				truncatedToolResult: ({ toolName }) => truncatedCallText(toolName),
				// A cut `replay: 'never'` call gets Flue's interrupted marker.
				interruptedToolResult: INTERRUPTED_TOOL_RESULT,
				// A cut answer goes on after Flue's two recovery signals.
				continueCutOff: { note: STREAM_RECOVERY_NOTES },
				...(durable
					? { recover: (ctx: RecoverContext) => this.recoverInput(ctx) }
					: this.options.recover
						? { recover: this.options.recover }
						: {}),
			},
			turn: {
				beforeFinish: (ctx) => this.beforeFinish(ctx),
				maxFinishCycles: Number.MAX_SAFE_INTEGER,
				onJoin: (ctx) => this.joined(ctx),
				...(canJoin ? { canJoin } : {}),
				...(modelErrors ? { onModelError: modelErrors } : {}),
			},
		});
		return host.open(
			harness,
			durable
				? { threadId: durable.threadId, logId: durable.logId }
				: { threadId: this.options.sessionId },
		);
	}

	/** Inputs joined the running turn: their messages go out before its next model call. */
	private async joined(ctx: JoinContext) {
		const run = this.run;
		for (const input of ctx.inputs) {
			const message =
				this.joining.get(input.inputId) ??
				fromModelMessage({ role: 'user', content: input.message });
			this.joining.delete(input.inputId);
			run?.pending.push(message);
			run?.joined.push(input.inputId);
		}
		const onJoin = this.options.onJoin;
		if (!onJoin) return undefined;
		if (!run) return onJoin(ctx);
		return this.guard(run, async () => onJoin(ctx));
	}

	private middleware(): ChatMiddleware {
		return {
			name: 'flue/agent-loop',
			onConfig: async (ctx, config) => {
				const run = this.run;
				if (!run || ctx.phase !== 'beforeModel') return;
				return this.guard(run, () =>
					this.beforeModelCall(run, config.messages, config.providerMessages ?? config.messages),
				);
			},
			onChunk: async (_ctx, chunk) => {
				const run = this.run;
				if (!run) return;
				if (run.assembler) await this.guard(run, () => this.modelChunk(run, chunk));
				else if (chunk.type === EventType.TOOL_CALL_RESULT)
					await this.guard(run, () => this.closeCutCall(run, chunk.toolCallId));
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
			const result = await work();
			await this.flushAdmissions(run);
			return result;
		} catch (error) {
			run.failure ??= { error };
			throw error;
		}
	}

	/**
	 * Before each model call: the loop's events, and Flue's request. On a
	 * binding, the call's messages (the harness transcript, with the joined
	 * steers) are the context. Without one, Flue's transcript is.
	 */
	private async beforeModelCall(
		run: LoopRun,
		transcript: ModelMessage[],
		callMessages: ModelMessage[],
	) {
		// A cut batch closes before the next model call, if no result chunk closed it.
		if (run.batch) await this.finishBatch(run, new Map());
		if (!run.firstCall) {
			if (run.lastTurn) await this.options.prepareNextTurn?.(run.lastTurn);
			if (run.pending.length === 0)
				run.pending = this.drainSteering().map((steer) => steer.message);
			await this.emit({ type: 'turn_start' });
		}
		// A model call after a failed one is a retry: the run goes on.
		if (run.stopped) this.state.errorMessage = undefined;
		run.stopped = false;
		run.firstCall = false;
		run.terminated = false;
		const pending = run.pending;
		run.pending = [];
		for (const message of pending) await this.emitMessage(message);
		const joined = run.joined;
		run.joined = [];
		if (joined.length > 0) await this.options.onJoined?.(joined);

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
				messages: this.options.durable ? [] : this.state.messages,
				tools: tools.map(toToolDeclaration),
			},
			info,
		);
		run.assembler = new AssistantStreamAssembler(info);
		run.assistantStarted = false;
		const perCall = {
			systemPrompts: request.systemPrompts,
			tools: tools.map((tool) => this.harnessTool(tool)),
			// pi sent the off value for `off` too (for example Anthropic `thinking.type: 'disabled'`).
			reasoning: { level: thinkingLevel, summary: true },
		};
		// TanStack keeps the mid-conversation record of each stored assistant message itself.
		if (this.options.durable) {
			const rewritten = await this.options.rewriteContext?.(transcript);
			if (rewritten) {
				this.placeReminders(run, rewritten.length);
				return {
					...perCall,
					messages: rewritten,
					providerMessages: this.withReminders(
						run,
						toModelContext(rewritten, info),
						rewritten.length,
					),
				};
			}
			this.placeReminders(run, transcript.length);
			return {
				...perCall,
				providerMessages: this.withReminders(
					run,
					toModelContext(callMessages, info),
					transcript.length,
				),
			};
		}
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
		return { ...perCall, providerMessages };
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
			run.pending = this.drainSteering().map((steer) => steer.message);
			return;
		}
		run.batch = this.createBatch(message);
	}

	private createBatch(message: AssistantMessage): ToolBatch {
		const calls = message.content.filter((block): block is ToolCall => block.type === 'toolCall');
		const tools = this.state.tools;
		return {
			assistant: message,
			calls,
			sequential: calls.some(
				(call) => tools.find((tool) => tool.name === call.name)?.executionMode === 'sequential',
			),
			truncated: message.stopReason === 'length',
			closed: new Set(),
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
		harnessTool = toHarnessTool(
			{
				...tool,
				async execute(toolCallId, args, signal) {
					const run = loop.run;
					if (!run?.batch) return tool.execute(toolCallId, args, signal);
					return loop.executeCall(run, run.batch, tool, toolCallId, args, signal);
				},
			},
			this.toolOptions(),
		);
		this.harnessTools.set(tool, harnessTool);
		return harnessTool;
	}

	private toolOptions(): HarnessToolOptions {
		const { onToolCall, onToolResult } = this.options;
		return {
			...(onToolCall ? { onCall: onToolCall } : {}),
			...(onToolResult ? { onResult: onToolResult } : {}),
		};
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
						result: errorToolResult(truncatedCallText(tool.name)),
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
				const text = batch.truncated
					? truncatedCallText(call.name)
					: skippedCallText(call, chatResults.get(call.id));
				outcome = { result: errorToolResult(text), isError: true };
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
		run.pending = this.drainSteering().map((steer) => steer.message);
	}

	/**
	 * The harness closed a call of a cut batch with an error result. The calls
	 * never run, so the batch ends when the last one is closed, with Flue's
	 * results and events.
	 */
	private async closeCutCall(run: LoopRun, toolCallId: string) {
		const batch = run.batch;
		if (!batch?.truncated) return;
		batch.closed.add(toolCallId);
		if (batch.calls.every((call) => batch.closed.has(call.id)))
			await this.finishBatch(run, new Map());
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

	/**
	 * The model stopped calling tools. Without a binding: continue the same
	 * run while steered messages wait. On a binding, the harness itself joins
	 * a steer that waits after the answer, and the `beforeFinish` option
	 * decides.
	 */
	private async beforeFinish(ctx: FinishContext) {
		const run = this.run;
		if (!run || run.stopped || run.failure || run.signal.aborted) return undefined;
		if (this.options.durable) {
			const beforeFinish = this.options.beforeFinish;
			if (!beforeFinish) return undefined;
			const added = await this.guard(run, () => beforeFinish(ctx));
			if (added?.ephemeral?.length)
				run.reminders.push({ messages: added.ephemeral, at: undefined });
			return added;
		}
		// pi polled the steering queue once, after `turn_end`.
		if (run.pending.length === 0) return undefined;
		return { messages: [{ role: 'user' as const, content: 'Continue.' }] };
	}

	/**
	 * A new reminder goes out at the end of this model call (the harness adds
	 * it), so it keeps that place in the later calls.
	 */
	private placeReminders(run: LoopRun, transcriptLength: number) {
		for (const reminder of run.reminders) reminder.at ??= transcriptLength;
	}

	/**
	 * The model call's messages with the reminders of earlier calls of the
	 * run, each at its place, as pi kept them in the transcript. The newest
	 * reminder is left out: the harness adds it to this call.
	 */
	private withReminders(run: LoopRun, messages: ModelMessage[], transcriptLength: number) {
		const placed = run.reminders.filter(
			(reminder): reminder is Reminder & { at: number } =>
				reminder.at !== undefined && reminder.at < transcriptLength,
		);
		if (placed.length === 0) return messages;
		// The transcript and the call's messages end the same way: count from the end.
		const offset = messages.length - transcriptLength;
		const result = messages.slice();
		for (const reminder of placed.toReversed())
			result.splice(Math.max(0, reminder.at + offset), 0, ...reminder.messages);
		return result;
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
