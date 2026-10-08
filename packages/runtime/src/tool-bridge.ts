/**
 * Flue tools as TanStack harness tools. Each `AgentTool` becomes a
 * `durableTool`: after a crash cuts a call, a `replay: 'safe'` tool runs again
 * (its finished steps return their stored values), and any other tool gives
 * the model a tool error.
 *
 * The session's tool wrappers keep the deadlines, the abort handling, and the
 * interceptor. The bridge adds what only the harness has: the call's durable
 * steps and the records staged with the tool batch.
 */
import { type JSONSchema, toolDefinition } from '@tanstack/ai';
import { type DurableToolContext, durableTool } from '@tanstack/ai-harness';
import type { HarnessLogRecord } from './conversation-records.ts';
import type { AgentTool, AgentToolResult, ImageContent, TextContent } from './llm-types.ts';
import { claimStepName, cloneStepValue } from './tool.ts';
import type { ToolStep } from './tool-types.ts';

/** One call of a bridged tool, before it runs. */
export interface HarnessToolCall {
	toolCallId: string;
	/** The call's durable steps, with Flue's step rules. */
	step: ToolStep;
	/** Stage host records. They land in the log with the tool batch, or never. */
	append(records: readonly HarnessLogRecord[]): void;
}

export interface HarnessToolOptions {
	/** Runs before the tool, so the session can give the call its step and its staged records. */
	onCall?: (call: HarnessToolCall) => void;
	/** Gets the tool's whole result: the content, `details`, and `terminate`. */
	onResult?: (toolCallId: string, result: AgentToolResult) => void;
}

/**
 * The TanStack tool for `tool`. The model sees the tool's content: one text
 * block as its exact text, and anything else as content parts.
 */
export function toHarnessTool(tool: AgentTool, options: HarnessToolOptions = {}) {
	const definition = toolDefinition({
		name: tool.name,
		description: tool.description,
		inputSchema: tool.parameters satisfies JSONSchema,
	});
	return durableTool(
		definition,
		async (args, context) => {
			const { toolCallId } = context;
			if (toolCallId === undefined)
				throw new Error(`[flue] Tool "${tool.name}" ran without a tool call id.`);
			options.onCall?.({
				toolCallId,
				step: toFlueStep(tool.name, context.step),
				append: context.append,
			});
			const result = await tool.execute(toolCallId, args, context.abortSignal);
			options.onResult?.(toolCallId, result);
			return toModelOutput(result.content);
		},
		{ replay: tool.replay ?? 'never' },
	);
}

/** TanStack's step with Flue's rules: a non-empty name used once per call, and a JSON value. */
function toFlueStep(toolName: string, step: DurableToolContext['step']) {
	const used = new Set<string>();
	return {
		do: <T>(name: string, fn: () => T | Promise<T>) => {
			const stepName = claimStepName(name, toolName, used);
			// `cloneStepValue` returns the JSON clone of the step's own value.
			return step.do(stepName, async () => cloneStepValue(await fn(), toolName, stepName) as T);
		},
	};
}

/**
 * The tool's return value for TanStack. TanStack parses a string result as
 * JSON when it can, so one text block goes out as a JSON string: the parse
 * gives back the exact text.
 */
function toModelOutput(content: readonly (TextContent | ImageContent)[]) {
	const [only] = content;
	if (content.length === 1 && only?.type === 'text') return JSON.stringify(only.text);
	return content.map((block) =>
		block.type === 'text'
			? { type: 'text' as const, content: block.text }
			: {
					type: 'image' as const,
					source: {
						type: 'data' as const,
						value: block.data,
						mimeType: block.mimeType,
					},
				},
	);
}
