import { EventType, type StepFinishedEvent, type StreamChunk } from '@tanstack/ai';
import { hashToolCallId, transformMessagesForReplay } from '@tanstack/ai/adapter-internals';
import { describe, expect, it } from 'vitest';
import type { DocumentContextBlock } from './document-attachments.ts';
import type { AgentMessage, AssistantMessage, ToolResultMessage } from './llm-types.ts';
import {
	type AssistantBlockEvent,
	AssistantStreamAssembler,
	type FlueModelInfo,
	fromModelMessage,
	toFlueUsage,
	toModelRequest,
} from './model-messages.ts';

const claude: FlueModelInfo = {
	provider: 'anthropic',
	api: 'anthropic-messages',
	id: 'claude-sonnet-4-5',
	input: ['text', 'image', 'document'],
	cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
};
const gpt: FlueModelInfo = {
	provider: 'openai',
	api: 'openai-responses',
	id: 'gpt-5',
	input: ['text', 'image'],
};
const textOnly: FlueModelInfo = {
	provider: 'mistral',
	api: 'mistral-conversations',
	id: 'mistral-small',
	input: ['text'],
};

const run = { threadId: 'thread', runId: 'run' };
const quotePdf: DocumentContextBlock = {
	type: 'image',
	data: 'cGRm',
	mimeType: 'application/pdf',
	filename: 'quote.pdf',
};

function assistant(
	content: AssistantMessage['content'],
	overrides: Partial<AssistantMessage> = {},
): AssistantMessage {
	return {
		role: 'assistant',
		content,
		api: claude.api,
		provider: claude.provider,
		model: claude.id,
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: 'stop',
		timestamp: 1,
		...overrides,
	};
}

function toolResult(
	content: ToolResultMessage['content'],
	overrides: Partial<ToolResultMessage> = {},
): ToolResultMessage {
	return {
		role: 'toolResult',
		toolCallId: 'call_1',
		toolName: 'read',
		content,
		isError: false,
		timestamp: 2,
		...overrides,
	};
}

/** OpenAI's adapter puts the reasoning signature on `STEP_FINISHED`, a field the AG-UI type does not list. */
function signedStep(event: StepFinishedEvent, signature: string) {
	return Object.assign(event, { signature });
}

/** Pushes every chunk, and copies each event's block when the event fires (the partial message keeps changing). */
function assemble(info: FlueModelInfo, chunks: StreamChunk[]) {
	const assembler = new AssistantStreamAssembler(info);
	const events = chunks.flatMap((chunk) =>
		assembler.push(chunk).map((event) => ({
			event,
			block: structuredClone(event.partial.content[event.contentIndex]),
		})),
	);
	return { events, message: assembler.finish() };
}

function eventTypes(events: { event: AssistantBlockEvent }[]) {
	return events.map(({ event }) => `${event.type}:${event.contentIndex}`);
}

function messagesFor(messages: AgentMessage[], target: FlueModelInfo) {
	return toModelRequest({ systemPrompt: 'Be brief.', messages, tools: [] }, target).messages;
}

describe('AssistantStreamAssembler', () => {
	it('builds thinking, text, and tool call blocks with their signatures and usage', () => {
		const { events, message } = assemble(claude, [
			{ type: EventType.REASONING_START, messageId: 'r1' },
			{
				type: EventType.REASONING_MESSAGE_START,
				messageId: 'r1',
				role: 'reasoning',
			},
			{ type: EventType.STEP_STARTED, stepName: 's1' },
			{
				type: EventType.REASONING_MESSAGE_CONTENT,
				messageId: 'r1',
				delta: 'Let me look.',
			},
			{
				type: EventType.REASONING_ENCRYPTED_VALUE,
				subtype: 'message',
				entityId: 'r1',
				encryptedValue: 'sig-a',
			},
			{ type: EventType.REASONING_MESSAGE_END, messageId: 'r1' },
			{ type: EventType.REASONING_END, messageId: 'r1' },
			{
				type: EventType.TEXT_MESSAGE_START,
				messageId: 'm1',
				role: 'assistant',
			},
			{
				type: EventType.TEXT_MESSAGE_CONTENT,
				messageId: 'm1',
				delta: 'Reading ',
			},
			{ type: EventType.TEXT_MESSAGE_CONTENT, messageId: 'm1', delta: 'it.' },
			{ type: EventType.TEXT_MESSAGE_END, messageId: 'm1' },
			{
				type: 'TOOL_CALL_START',
				toolCallId: 'call_1',
				toolCallName: 'read',
				metadata: { thoughtSignature: 'ts-1' },
			},
			{
				type: EventType.TOOL_CALL_ARGS,
				toolCallId: 'call_1',
				delta: '{"path":',
			},
			{
				type: EventType.TOOL_CALL_ARGS,
				toolCallId: 'call_1',
				delta: '"a.txt"}',
			},
			{ type: 'TOOL_CALL_END', toolCallId: 'call_1' },
			{
				type: EventType.RUN_FINISHED,
				...run,
				finishReason: 'tool_calls',
				responseId: 'resp_1',
				usage: {
					promptTokens: 1_000_000,
					completionTokens: 100_000,
					totalTokens: 1_100_000,
					promptTokensDetails: {
						cachedTokens: 400_000,
						cacheWriteTokens: 100_000,
					},
					completionTokensDetails: { reasoningTokens: 20_000 },
				},
			},
		]);

		expect(eventTypes(events)).toEqual([
			'thinking_start:0',
			'thinking_delta:0',
			'thinking_end:0',
			'text_start:1',
			'text_delta:1',
			'text_delta:1',
			'text_end:1',
			'toolcall_delta:2',
			'toolcall_delta:2',
			'toolcall_end:2',
		]);
		expect(message.content).toEqual([
			{
				type: 'thinking',
				thinking: 'Let me look.',
				thinkingSignature: 'sig-a',
			},
			{ type: 'text', text: 'Reading it.' },
			{
				type: 'toolCall',
				id: 'call_1',
				name: 'read',
				arguments: { path: 'a.txt' },
				thoughtSignature: 'ts-1',
			},
		]);
		expect(message).toMatchObject({
			stopReason: 'toolUse',
			rawStopReason: 'tool_calls',
			responseId: 'resp_1',
		});
		expect(message.usage).toEqual({
			input: 500_000,
			output: 100_000,
			cacheRead: 400_000,
			cacheWrite: 100_000,
			reasoning: 20_000,
			totalTokens: 1_100_000,
			cost: {
				input: expect.closeTo(1.5, 9),
				output: expect.closeTo(1.5, 9),
				cacheRead: expect.closeTo(0.12, 9),
				cacheWrite: expect.closeTo(0.375, 9),
				total: expect.closeTo(3.495, 9),
			},
		});
	});

	it('ends a thinking block after a signature that arrives after its end event', () => {
		const { events } = assemble(gpt, [
			{
				type: EventType.REASONING_MESSAGE_START,
				messageId: 'r1',
				role: 'reasoning',
			},
			{ type: EventType.STEP_STARTED, stepName: 's1' },
			{
				type: EventType.REASONING_MESSAGE_CONTENT,
				messageId: 'r1',
				delta: 'Plan.',
			},
			{ type: EventType.REASONING_MESSAGE_END, messageId: 'r1' },
			signedStep({ type: EventType.STEP_FINISHED, stepName: 's1' }, 'enc-1'),
			{ type: EventType.TEXT_MESSAGE_CONTENT, messageId: 'm1', delta: 'Done.' },
		]);

		const thinkingEnd = events.find(({ event }) => event.type === 'thinking_end');
		expect(thinkingEnd?.block).toEqual({
			type: 'thinking',
			thinking: 'Plan.',
			thinkingSignature: 'enc-1',
		});
		expect(eventTypes(events)).toEqual([
			'thinking_start:0',
			'thinking_delta:0',
			'thinking_end:0',
			'text_start:1',
			'text_delta:1',
		]);
	});

	it('keeps redacted thinking as its encrypted payload', () => {
		const { events, message } = assemble(claude, [
			{
				type: EventType.REASONING_MESSAGE_START,
				messageId: 'redacted_thinking-1',
				role: 'reasoning',
			},
			{
				type: EventType.REASONING_ENCRYPTED_VALUE,
				subtype: 'message',
				entityId: 'redacted_thinking-1',
				encryptedValue: 'opaque',
			},
			{
				type: EventType.REASONING_MESSAGE_END,
				messageId: 'redacted_thinking-1',
			},
		]);

		expect(message.content).toEqual([
			{
				type: 'thinking',
				thinking: '[Reasoning redacted]',
				thinkingSignature: 'opaque',
				redacted: true,
			},
		]);
		expect(eventTypes(events)).toEqual(['thinking_start:0']);
	});

	it.each([
		{
			chunk: { type: EventType.RUN_FINISHED, ...run, finishReason: 'stop' },
			stopReason: 'stop',
			errorMessage: undefined,
		},
		{
			chunk: { type: EventType.RUN_FINISHED, ...run, finishReason: 'length' },
			stopReason: 'length',
			errorMessage: undefined,
		},
		{
			chunk: {
				type: EventType.RUN_FINISHED,
				...run,
				finishReason: 'tool_calls',
			},
			stopReason: 'toolUse',
			errorMessage: undefined,
		},
		{
			chunk: {
				type: EventType.RUN_FINISHED,
				...run,
				finishReason: 'content_filter',
			},
			stopReason: 'error',
			errorMessage: 'Provider finish_reason: content_filter',
		},
		{
			chunk: { type: EventType.RUN_ERROR, message: 'Overloaded' },
			stopReason: 'error',
			errorMessage: 'Overloaded',
		},
		{
			chunk: {
				type: EventType.RUN_ERROR,
				message: 'Request aborted',
				code: 'aborted',
			},
			stopReason: 'aborted',
			errorMessage: 'Request aborted',
		},
	] satisfies {
		chunk: StreamChunk;
		stopReason: AssistantMessage['stopReason'];
		errorMessage: string | undefined;
	}[])(
		'maps $chunk.type $chunk.finishReason$chunk.code to $stopReason',
		({ chunk, stopReason, errorMessage }) => {
			const { message } = assemble(claude, [chunk]);
			expect(message.stopReason).toBe(stopReason);
			expect(message.errorMessage).toBe(errorMessage);
		},
	);
});

describe('toModelRequest', () => {
	it('maps each assistant block, keeps the block order, and tags the source', () => {
		const [message] = messagesFor(
			[
				assistant(
					[
						{
							type: 'thinking',
							thinking: 'Check first.',
							thinkingSignature: 'sig-a',
						},
						{
							type: 'toolCall',
							id: 'call_1',
							name: 'read',
							arguments: { path: 'a.txt' },
							thoughtSignature: 'ts-1',
						},
						{ type: 'text', text: 'Read it.' },
					],
					{ stopReason: 'error' },
				),
			],
			claude,
		);

		expect(message).toEqual({
			role: 'assistant',
			content: 'Read it.',
			thinking: [{ content: 'Check first.', signature: 'sig-a' }],
			toolCalls: [
				{
					id: 'call_1',
					type: 'function',
					function: { name: 'read', arguments: '{"path":"a.txt"}' },
					metadata: { thoughtSignature: 'ts-1' },
				},
			],
			blockOrder: [
				{ type: 'thinking', index: 0 },
				{ type: 'tool-call', id: 'call_1' },
				{ type: 'text', length: 8 },
			],
			metadata: {
				tanstack: {
					source: {
						provider: 'anthropic',
						api: 'anthropic-messages',
						model: 'claude-sonnet-4-5',
					},
					stopReason: 'error',
				},
			},
		});
	});

	it('renders signals as user text, and tool errors with their text', () => {
		const messages = messagesFor(
			[
				{
					role: 'signal',
					type: 'resource',
					content: 'a < b',
					attributes: { name: 'notes' },
					timestamp: 3,
				},
				toolResult([{ type: 'text', text: 'not found' }], { isError: true }),
			],
			claude,
		);

		expect(messages).toEqual([
			{
				role: 'user',
				content: '<signal type="resource" name="notes">\na &lt; b\n</signal>',
			},
			{
				role: 'tool',
				toolCallId: 'call_1',
				name: 'read',
				content: 'not found',
				error: 'not found',
			},
		]);
	});

	it('sends images and documents to a model that reads them', () => {
		const [message] = messagesFor(
			[
				{
					role: 'user',
					content: [{ type: 'image', data: 'aW1n', mimeType: 'image/png' }, quotePdf],
					timestamp: 1,
				},
			],
			claude,
		);

		expect(message?.content).toEqual([
			{
				type: 'image',
				source: { type: 'data', value: 'aW1n', mimeType: 'image/png' },
			},
			{
				type: 'document',
				source: { type: 'data', value: 'cGRm', mimeType: 'application/pdf' },
				metadata: { filename: 'quote.pdf' },
			},
		]);
	});

	it('replaces media that the model cannot read with placeholders', () => {
		const messages = messagesFor(
			[
				{
					role: 'user',
					content: [
						{ type: 'text', text: 'Look:' },
						{ type: 'image', data: 'YQ==', mimeType: 'image/png' },
						{ type: 'image', data: 'Yg==', mimeType: 'image/png' },
						quotePdf,
					],
					timestamp: 1,
				},
				toolResult([{ type: 'image', data: 'Yw==', mimeType: 'image/png' }]),
			],
			textOnly,
		);

		expect(messages).toEqual([
			{
				role: 'user',
				content: [
					{ type: 'text', content: 'Look:' },
					{
						type: 'text',
						content: '(image omitted: model does not support images)',
					},
					{
						type: 'text',
						content:
							'(document "quote.pdf" omitted: model API "mistral-conversations" does not support document input)',
					},
				],
			},
			{
				role: 'tool',
				toolCallId: 'call_1',
				name: 'read',
				content: '(tool image omitted: model does not support images)',
			},
		]);
	});
});

describe('a conversation that changes model', () => {
	const history: AgentMessage[] = [
		{ role: 'user', content: 'Read a.txt', timestamp: 1 },
		assistant([
			{
				type: 'thinking',
				thinking: 'Use the tool.',
				thinkingSignature: 'sig-a',
			},
			{
				type: 'toolCall',
				id: 'toolu|01',
				name: 'read',
				arguments: { path: 'a.txt' },
				thoughtSignature: 'ts-1',
			},
		]),
		toolResult([{ type: 'text', text: 'hello' }], { toolCallId: 'toolu|01' }),
	];
	const toolIdRule = (id: string) => `call_${hashToolCallId(id)}`;

	function replayFor(target: FlueModelInfo) {
		const source = {
			provider: target.provider,
			api: target.api,
			model: target.id,
		};
		return transformMessagesForReplay(messagesFor(history, target), source, toolIdRule).messages;
	}

	it('sends no foreign signatures and gives the tool call a new id after the switch', () => {
		const [, call, result] = replayFor(gpt);
		const newId = `call_${hashToolCallId('toolu|01')}`;

		expect(call).toMatchObject({
			content: 'Use the tool.',
			toolCalls: [{ id: newId }],
		});
		expect(call?.toolCalls?.[0]?.metadata).toEqual({});
		expect(call?.thinking).toBeUndefined();
		expect(result).toMatchObject({ role: 'tool', toolCallId: newId });
	});

	it('keeps the signatures for the model that made them', () => {
		const [, call, result] = replayFor(claude);

		expect(call?.thinking).toEqual([{ content: 'Use the tool.', signature: 'sig-a' }]);
		expect(call?.toolCalls).toEqual([
			{
				id: 'toolu|01',
				type: 'function',
				function: { name: 'read', arguments: '{"path":"a.txt"}' },
				metadata: { thoughtSignature: 'ts-1' },
			},
		]);
		expect(result).toMatchObject({ toolCallId: 'toolu|01' });
	});
});

describe('fromModelMessage', () => {
	it('gives back the Flue messages that toModelRequest sent', () => {
		const sent: AgentMessage[] = [
			{
				role: 'user',
				content: [
					{ type: 'text', text: 'See the files.' },
					{ type: 'image', data: 'aW1n', mimeType: 'image/png' },
					quotePdf,
				],
				timestamp: 1,
			},
			assistant([
				{
					type: 'thinking',
					thinking: 'Check first.',
					thinkingSignature: 'sig-a',
				},
				{
					type: 'thinking',
					thinking: '[Reasoning redacted]',
					thinkingSignature: 'opaque',
					redacted: true,
				},
				{
					type: 'toolCall',
					id: 'call_1',
					name: 'read',
					arguments: { path: 'a.txt' },
					thoughtSignature: 'ts-1',
				},
				{ type: 'text', text: 'Reading it.' },
			]),
			toolResult([{ type: 'text', text: 'not found' }], { isError: true }),
		];

		const received = messagesFor(sent, claude).map((message) => fromModelMessage(message));

		expect(received).toMatchObject([
			{
				role: 'user',
				content: sent[0]?.role === 'user' ? sent[0].content : [],
			},
			{
				role: 'assistant',
				api: 'anthropic-messages',
				provider: 'anthropic',
				model: 'claude-sonnet-4-5',
				stopReason: 'toolUse',
				content: [
					{
						type: 'thinking',
						thinking: 'Check first.',
						thinkingSignature: 'sig-a',
					},
					{
						type: 'thinking',
						thinking: '[Reasoning redacted]',
						thinkingSignature: 'opaque',
						redacted: true,
					},
					{
						type: 'toolCall',
						id: 'call_1',
						name: 'read',
						arguments: { path: 'a.txt' },
						thoughtSignature: 'ts-1',
					},
					{ type: 'text', text: 'Reading it.' },
				],
			},
			{
				role: 'toolResult',
				toolCallId: 'call_1',
				toolName: 'read',
				content: [{ type: 'text', text: 'not found' }],
				isError: true,
			},
		]);
	});

	it('keeps the stop reason of a failed turn', () => {
		const [message] = messagesFor(
			[
				assistant([{ type: 'text', text: 'Partial' }], {
					stopReason: 'aborted',
				}),
			],
			claude,
		);

		expect(message && fromModelMessage(message)).toMatchObject({
			role: 'assistant',
			stopReason: 'aborted',
		});
	});
});

describe('toFlueUsage', () => {
	// Anthropic's own split: 1.6M prompt tokens, of which 200k are read from the
	// cache and 400k are cache writes (100k of them with a 1 hour retention).
	const usage = {
		promptTokens: 1_600_000,
		completionTokens: 100_000,
		totalTokens: 1_700_000,
		promptTokensDetails: {
			cachedTokens: 200_000,
			cacheWriteTokens: 400_000,
			cacheWrite1hTokens: 100_000,
		},
	};

	it('counts input without the cached tokens and prices each part', () => {
		const flue = toFlueUsage(usage, claude.cost);

		expect(flue).toMatchObject({
			input: 1_000_000,
			output: 100_000,
			cacheRead: 200_000,
			cacheWrite: 400_000,
			cacheWrite1h: 100_000,
			totalTokens: 1_700_000,
		});
		// $3 input, $1.50 output, $0.06 cache read, and cache writes of
		// 300k at $3.75 ($1.125) plus 100k at 2x input ($0.60).
		expect(flue.cost.input).toBeCloseTo(3);
		expect(flue.cost.output).toBeCloseTo(1.5);
		expect(flue.cost.cacheRead).toBeCloseTo(0.06);
		expect(flue.cost.cacheWrite).toBeCloseTo(1.725);
		expect(flue.cost.total).toBeCloseTo(6.285);
	});

	it('costs nothing for a model with no prices', () => {
		expect(toFlueUsage(usage, undefined).cost.total).toBe(0);
	});
});
