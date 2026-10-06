import { chat } from '@tanstack/ai';
import { getModels, getProviders } from '@tanstack/ai-models';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
	type DocumentContextBlock,
	documentOmittedPlaceholder,
	mergeOperationAttachments,
	toPublicAttachment,
} from './document-attachments.ts';
import { init, useModel } from './index.ts';
import type { AgentMessage } from './llm-types.ts';
import { modelInfo, toModelRequest } from './model-messages.ts';
import { sqlite, start } from './node/index.ts';
import { createModelAdapter } from './providers/adapters.ts';
import { builtinProvider } from './providers/builtins.ts';
import { parseDeliveredMessage } from './runtime/schemas.ts';
import {
	type FauxContext,
	fauxAssistantMessage,
	fauxProvider,
	fauxText,
} from './test-utils/faux.ts';

const PDF = 'JVBERi0xLjQK';
const PNG = 'iVBORw0KGgo=';

afterEach(() => {
	vi.unstubAllGlobals();
});

const quotePdf: DocumentContextBlock = {
	type: 'image',
	data: PDF,
	mimeType: 'application/pdf',
	filename: 'quote.pdf',
};

function documentMessages(): AgentMessage[] {
	return [
		{
			role: 'user',
			timestamp: 0,
			content: [
				{ type: 'text', text: 'Summarize this quote.' },
				{ type: 'image', data: PNG, mimeType: 'image/png' },
				quotePdf,
			],
		},
	];
}

function catalogModel(providerId: string, modelId: string) {
	const record = getProviders().find((provider) => provider.id === providerId);
	if (!record) throw new Error(`No provider "${providerId}".`);
	const provider = builtinProvider({
		provider: record,
		models: getModels(providerId),
	});
	const model = provider.getModels().find((entry) => entry.id === modelId);
	if (!model) throw new Error(`No model "${providerId}/${modelId}".`);
	return { provider, model };
}

/** The JSON body that the provider adapter sends for `messages`, read at the `fetch` boundary. */
async function sentBody(providerId: string, modelId: string) {
	const bodies: Record<string, unknown>[] = [];
	vi.stubGlobal('fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
		bodies.push(JSON.parse(await new Request(input, init).text()));
		return Response.json({ error: { message: 'test stop' } }, { status: 400 });
	});
	const { provider, model } = catalogModel(providerId, modelId);
	const request = toModelRequest(
		{ systemPrompt: 'system', messages: documentMessages(), tools: [] },
		modelInfo(model),
	);
	const adapter = createModelAdapter(provider, model, {
		auth: { auth: { apiKey: 'test-key' } },
		promptCacheKey: 'conv-1',
	});
	try {
		for await (const _chunk of chat({
			adapter,
			messages: request.messages,
			systemPrompts: request.systemPrompts,
		})) {
			// Drain the stream.
		}
	} catch {
		// The stub answers 400.
	}
	const [body] = bodies;
	if (!body) throw new Error('The adapter sent no request.');
	return body;
}

describe('provider requests', () => {
	it('sends documents to Anthropic as native document blocks', async () => {
		const body = await sentBody('anthropic', 'claude-sonnet-4-5');
		const [message] = body.messages as Array<{
			content: Array<Record<string, unknown>>;
		}>;
		expect(message?.content[1]).toMatchObject({
			type: 'image',
			source: { type: 'base64', media_type: 'image/png', data: PNG },
		});
		expect(message?.content[2]).toMatchObject({
			type: 'document',
			title: 'quote.pdf',
			source: { type: 'base64', media_type: 'application/pdf', data: PDF },
		});
	});

	it('sends documents to OpenAI Responses as input_file parts', async () => {
		const body = await sentBody('openai', 'gpt-5');
		const user = (body.input as Array<{ role?: string; content?: unknown }>).find(
			(item) => item.role === 'user',
		);
		const content = user?.content as Array<Record<string, unknown>>;
		expect(content[1]).toMatchObject({ type: 'input_image' });
		expect(content[2]).toEqual({
			type: 'input_file',
			filename: 'quote.pdf',
			file_data: `data:application/pdf;base64,${PDF}`,
		});
	});

	it('sends documents to Google as inlineData', async () => {
		const body = await sentBody('google', 'gemini-2.5-flash');
		const [content] = body.contents as Array<{
			parts: Array<Record<string, unknown>>;
		}>;
		expect(content?.parts[2]).toEqual({
			inlineData: { mimeType: 'application/pdf', data: PDF },
		});
	});

	it('replaces documents with a placeholder for unsupported APIs', () => {
		vi.spyOn(console, 'warn').mockImplementation(() => {});
		const { model } = catalogModel('amazon-bedrock', 'amazon.nova-2-lite-v1:0');
		const [message] = toModelRequest(
			{ systemPrompt: 'system', messages: documentMessages(), tools: [] },
			modelInfo(model),
		).messages;
		expect(message?.content).toEqual([
			{ type: 'text', content: 'Summarize this quote.' },
			{
				type: 'image',
				source: { type: 'data', value: PNG, mimeType: 'image/png' },
			},
			{
				type: 'text',
				content: documentOmittedPlaceholder('bedrock-converse-stream', 'quote.pdf'),
			},
		]);
	});

	it('leaves images as they are without documents', () => {
		const { model } = catalogModel('anthropic', 'claude-sonnet-4-5');
		const [message] = toModelRequest(
			{
				systemPrompt: '',
				messages: [
					{
						role: 'user',
						timestamp: 0,
						content: [{ type: 'image', data: PNG, mimeType: 'image/png' }],
					},
				],
				tools: [],
			},
			modelInfo(model),
		).messages;
		expect(message?.content).toEqual([
			{
				type: 'image',
				source: { type: 'data', value: PNG, mimeType: 'image/png' },
			},
		]);
	});
});

describe('attachment shapes', () => {
	it('folds documents into the image carrier and back', () => {
		const merged = mergeOperationAttachments(
			[{ type: 'image', data: PNG, mimeType: 'image/png' }],
			[{ type: 'document', data: PDF, mimeType: 'application/pdf' }],
		);
		expect(merged).toEqual([
			{ type: 'image', data: PNG, mimeType: 'image/png' },
			{ type: 'image', data: PDF, mimeType: 'application/pdf' },
		]);
		expect(merged?.map(toPublicAttachment).map((block) => block.type)).toEqual([
			'image',
			'document',
		]);
	});

	it('rejects unsupported document MIME types', () => {
		expect(() =>
			mergeOperationAttachments(undefined, [
				{ type: 'document', data: PDF, mimeType: 'application/msword' },
			]),
		).toThrow(/Unsupported document mimeType/);
		let rejection: unknown;
		try {
			parseDeliveredMessage({
				kind: 'user',
				body: 'hi',
				attachments: [{ type: 'document', data: PDF, mimeType: 'text/html' }],
			});
		} catch (error) {
			rejection = error;
		}
		expect(rejection).toMatchObject({
			details: 'Document mimeType must be one of: application/pdf.',
		});
	});

	it('accepts document attachments on the wire', () => {
		expect(
			parseDeliveredMessage({
				kind: 'user',
				body: 'hi',
				attachments: [
					{ type: 'image', data: PNG, mimeType: 'image/png' },
					{
						type: 'document',
						data: PDF,
						mimeType: 'application/pdf',
						filename: 'quote.pdf',
					},
				],
			}),
		).toMatchObject({ attachments: [{ type: 'image' }, { type: 'document' }] });
	});
});

describe('end to end', () => {
	async function runWithDocument(api: string) {
		function DocumentAgent() {
			useModel('faux/model', { compaction: false });
			return 'Reply to the user.';
		}
		const faux = fauxProvider({
			api,
			models: [{ id: 'model', input: ['text', 'image'] }],
		});
		const seen: FauxContext[] = [];
		faux.setResponses([
			(context) => {
				seen.push(context);
				return fauxAssistantMessage([fauxText('Summarized.')], {
					stopReason: 'stop',
				});
			},
		]);
		const runtime = await start({
			agents: [DocumentAgent],
			db: sqlite(),
			providers: [faux.provider],
			env: {},
		});
		const agent = init(DocumentAgent, { id: `document-${api}` });
		try {
			await expect(
				agent.read(
					await agent.dispatch({
						message: {
							kind: 'user',
							body: 'Summarize this quote.',
							attachments: [
								{
									type: 'document',
									data: PDF,
									mimeType: 'application/pdf',
									filename: 'quote.pdf',
								},
							],
						},
					}),
				),
			).resolves.toMatchObject({ text: 'Summarized.' });
		} finally {
			await agent.abort();
			await runtime.stop();
		}
		const [request] = seen;
		if (!request) throw new Error('model was not called');
		const user = request.messages.find((message) => message.role === 'user');
		if (user?.role !== 'user' || !Array.isArray(user.content))
			throw new Error('user message missing');
		return user.content;
	}

	it('persists a delivered document and forwards it as a native document', async () => {
		const content = await runWithDocument('anthropic-messages');
		const text = content.find((block) => block.type === 'text');
		expect(text?.type === 'text' && text.text).toMatch(
			/<document id="[^"]+" mimeType="application\/pdf" filename="quote\.pdf" \/>/,
		);
		expect(content).toContainEqual(quotePdf);
	});

	it('replaces a delivered document with a placeholder on an unsupported API', async () => {
		vi.spyOn(console, 'warn').mockImplementation(() => {});
		const content = await runWithDocument('faux');
		expect(content).toContainEqual({
			type: 'text',
			text: documentOmittedPlaceholder('faux', 'quote.pdf'),
		});
		expect(content.some((block) => block.type === 'image')).toBe(false);
	});
});
