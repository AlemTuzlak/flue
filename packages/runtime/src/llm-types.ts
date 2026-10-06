/**
 * The message, content, usage, and tool shapes that Flue stores and passes
 * around. They are Flue's own types: the conversation records and the public
 * API use them, and the model layer converts them at its boundary.
 */

/** How hard the model thinks. `off` sends no reasoning request. */
export type ThinkingLevel = 'off' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'max';

type JsonValue = null | boolean | number | string | readonly JsonValue[] | JsonObject;

export type JsonObject = { [key: string]: JsonValue };

export interface TextContent {
	type: 'text';
	text: string;
	/** Provider data that lets the same provider replay this text block. */
	textSignature?: string;
}

export interface ThinkingContent {
	type: 'thinking';
	thinking: string;
	/** Provider signature (or the encrypted payload of redacted thinking) for replay. */
	thinkingSignature?: string;
	/** True when the provider redacted the thinking. The payload is in `thinkingSignature`. */
	redacted?: boolean;
}

export interface ImageContent {
	type: 'image';
	/** Base64 bytes. */
	data: string;
	mimeType: string;
}

export interface ToolCall {
	type: 'toolCall';
	id: string;
	name: string;
	arguments: JsonObject;
	/** Gemini's signature for this call, for replay to the same model. */
	thoughtSignature?: string;
	/** OpenAI Responses namespace of a tool that was added in mid-conversation. */
	namespace?: string;
}

export interface Usage {
	input: number;
	output: number;
	cacheRead: number;
	cacheWrite: number;
	/** The part of `cacheWrite` written with a 1 hour retention, when the provider reports it. */
	cacheWrite1h?: number;
	/** Reasoning tokens, when the provider reports them. A part of `output`. */
	reasoning?: number;
	totalTokens: number;
	cost: {
		input: number;
		output: number;
		cacheRead: number;
		cacheWrite: number;
		total: number;
	};
}

export type StopReason =
	'pending' | 'stop' | 'length' | 'toolUse' | 'error' | 'aborted' | 'deferred';

interface DiagnosticErrorInfo {
	name?: string;
	message: string;
	stack?: string;
	code?: string | number;
}

/** Structured data that a provider boundary attaches to an assistant message. */
export interface AssistantMessageDiagnostic {
	type: string;
	timestamp: number;
	error?: DiagnosticErrorInfo;
	details?: JsonObject;
}

export interface UserMessage {
	role: 'user';
	content: string | (TextContent | ImageContent)[];
	timestamp: number;
}

export interface AssistantMessage {
	role: 'assistant';
	content: (TextContent | ThinkingContent | ToolCall)[];
	/** The wire API that produced the message, for example `anthropic-messages`. */
	api: string;
	provider: string;
	model: string;
	responseModel?: string;
	responseId?: string;
	/** The provider's own effort value that this response used, when known. */
	providerThinkingLevel?: string;
	diagnostics?: AssistantMessageDiagnostic[];
	usage: Usage;
	stopReason: StopReason;
	errorMessage?: string;
	rawStopReason?: string;
	endTurn?: boolean;
	timestamp: number;
}

export interface ToolResultMessage<TDetails = JsonValue> {
	role: 'toolResult';
	toolCallId: string;
	toolName: string;
	content: (TextContent | ImageContent)[];
	details?: TDetails;
	/** Usage of the tool's own execution, when it reports any. */
	usage?: Usage;
	isError: boolean;
	timestamp: number;
}

/** A tool as the model sees it: name, description, and JSON Schema parameters. */
export interface ToolDeclaration {
	name: string;
	description: string;
	parameters: object;
}

/** System instructions and tool changes at one point in the transcript. */
export interface SystemMessage {
	role: 'system';
	content: string | TextContent[];
	sections?: Record<string, string | null>;
	toolsAdded?: ToolDeclaration[];
	toolsRemoved?: { name: string }[];
	timestamp: number;
}

/** A Flue signal: a tagged notice for the model, such as a resource change. */
export interface SignalMessage {
	role: 'signal';
	type: string;
	tagName?: string;
	content: string;
	attributes?: Record<string, string>;
	timestamp: number;
}

type Message = SystemMessage | UserMessage | AssistantMessage | ToolResultMessage;

/** Every message an agent conversation holds. */
export type AgentMessage = Message | SignalMessage;

/** What a tool returns: content for the model, plus structured details. */
export interface AgentToolResult<TDetails = JsonValue | undefined> {
	content: (TextContent | ImageContent)[];
	details: TDetails;
	usage?: Usage;
	/** Stop after this tool batch. The loop stops only when every result of the batch sets it. */
	terminate?: boolean;
}

type AgentToolUpdateCallback<TDetails = any> = (partialResult: AgentToolResult<TDetails>) => void;

type ToolExecutionMode = 'sequential' | 'parallel';

/**
 * A model-callable tool. `TParams` is the type of the arguments that
 * `execute` receives.
 */
export interface AgentTool<TParams = any, TDetails = any> extends ToolDeclaration {
	label: string;
	/** Changes the raw arguments before the schema check. */
	prepareArguments?: (args: unknown) => TParams;
	/**
	 * Runs the tool. Throw on failure instead of encoding the error in `content`.
	 * A method, so a tool with typed params fits a list of `AgentTool<any>`.
	 */
	execute(
		toolCallId: string,
		params: TParams,
		signal?: AbortSignal,
		onUpdate?: AgentToolUpdateCallback<TDetails>,
	): Promise<AgentToolResult<TDetails>>;
	/** What happens to a call that a crash cut: `safe` runs it again, `never` gives the model an error. */
	replay?: 'never' | 'safe';
	/** `sequential` makes the whole tool batch run one call at a time. */
	executionMode?: ToolExecutionMode;
}
