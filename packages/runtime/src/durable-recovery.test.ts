import { describe, expect, it } from 'vitest';
import { createFlueContext } from './client.ts';
import type { HarnessLogRecord } from './conversation-records.ts';
import { buildConversationContext } from './conversation-reducer.ts';
import { ConversationRecordWriter } from './conversation-writer.ts';
import { type AgentFunction, useModel, useSubagent, useTool } from './index.ts';
import { sqlite } from './node/agent-execution-store.ts';
import { setProvider } from './providers/registry.ts';
import {
	type AgentSubmissionInput,
	createAgentSubmissionSessionHandler,
	createDirectAgentSubmissionInput,
	ensureInstanceIdentity,
} from './runtime/agent-submissions.ts';
import { InMemoryAttachmentStore } from './runtime/attachment-store.ts';
import type { ConversationStreamStore } from './runtime/conversation-stream-store.ts';
import { harnessLogRecordsOf } from './runtime/harness-log-store.ts';
import {
	createHostRecordAppend,
	createInstanceHarnessHost,
} from './runtime/instance-harness-host.ts';
import { resolveModel } from './runtime/providers.ts';
import {
	type FauxContext,
	type FauxResponseStep,
	fauxAssistantMessage,
	fauxProvider,
	fauxText,
	fauxToolCall,
} from './test-utils/faux.ts';

const instanceId = 'instance-1';
const LEASE_MS = 60_000;

const INTERRUPTED_TEXT =
	'{"type":"interrupted","message":"Tool execution was interrupted before completion. The outcome is unknown."}';
const INTERRUPTED_NOTE =
	'<signal type="stream_interrupted">\nThe previous assistant stream was interrupted.\n</signal>';
const CONTINUED_NOTE =
	'<signal type="stream_continued">\nContinue from the durable partial assistant response.\n</signal>';

function pathOf(agentName: string) {
	return `agents/${agentName}/${instanceId}`;
}

async function openStores() {
	const adapter = sqlite();
	await adapter.migrate?.();
	const connection = await adapter.connect();
	return {
		streams: connection.conversationStreamStore,
		submissions: connection.submissionStore,
	};
}

/** A Flue context whose sessions run on a new instance host over `streams`. */
async function hostContext(agentName: string, streams: ConversationStreamStore, ownerId: string) {
	const identity = { agentName, instanceId };
	const path = pathOf(agentName);
	const host = createInstanceHarnessHost({ streams, path, identity, ownerId });
	const binding = { host, logId: path };
	const writer = await ConversationRecordWriter.overHarness({
		store: streams,
		path,
		identity,
		append: createHostRecordAppend(binding),
	});
	const ctx = createFlueContext({
		id: instanceId,
		agentName,
		env: {},
		agentConfig: { resolveModel },
		conversationWriter: writer,
		attachmentStore: new InMemoryAttachmentStore(),
		harnessHost: binding,
	});
	return { ctx, host, writer };
}

function runSubmission(
	agent: AgentFunction,
	input: AgentSubmissionInput,
	ctx: ReturnType<typeof createFlueContext>,
	attemptId: string,
) {
	return createAgentSubmissionSessionHandler(agent, input, (session) =>
		session.processSubmissionInput(input, {
			submissionAttempt: { submissionId: input.submissionId, attemptId },
		}),
	)(ctx);
}

/**
 * Run one submission of `agent` on host A until `stopPoint` resolves, then
 * stop host A as a crash does (`close({ recoverable: true })`). Host B opens
 * the same stores and runs the submission again: the harness recovers the
 * cut turn there.
 */
async function stopAndRecover(options: {
	agent: AgentFunction;
	responses: FauxResponseStep[];
	stopPoint: (streams: ConversationStreamStore) => Promise<void>;
	/** Stream each answer at this many tokens per second. */
	tokensPerSecond?: number;
	/** Runs when host A stops, before it closes. */
	onStop?: () => void;
}) {
	const { agent } = options;
	const agentName = agent.name;
	const faux = fauxProvider({
		models: [{ id: 'model' }],
		...(options.tokensPerSecond ? { tokensPerSecond: options.tokensPerSecond } : {}),
	});
	faux.setResponses(options.responses);
	setProvider(faux.provider);
	const { streams, submissions } = await openStores();
	const a = await hostContext(agentName, streams, 'host-a');
	const { conversationId } = await ensureInstanceIdentity(a.writer, agent, undefined);
	const input = await createDirectAgentSubmissionInput({
		agent: agentName,
		id: instanceId,
		message: { kind: 'user', body: 'Go.' },
	});
	// The stream store takes the records of a submission only from its
	// current attempt, so the submission has a row that host A claims.
	await submissions.admitDirect(input);
	await submissions.markSubmissionCanonicalReady(input.submissionId);
	const claimed = await submissions.claimSubmission({
		submissionId: input.submissionId,
		attemptId: 'attempt-1',
		ownerId: 'host-a',
		leaseExpiresAt: Date.now() + LEASE_MS,
	});
	if (!claimed) throw new Error('The submission was not claimed.');
	const runA = runSubmission(agent, input, a.ctx, 'attempt-1').catch(() => undefined);
	await options.stopPoint(streams);
	options.onStop?.();
	await a.host.close({ recoverable: true });
	await runA;

	const b = await hostContext(agentName, streams, 'host-b');
	const replaced = await submissions.replaceSubmissionAttempt(
		{ submissionId: input.submissionId, attemptId: 'attempt-1' },
		'attempt-2',
		{ ownerId: 'host-b', leaseExpiresAt: Date.now() + LEASE_MS },
	);
	if (!replaced) throw new Error('The submission has no replacement attempt.');
	await runSubmission(agent, input, b.ctx, 'attempt-2');
	await b.host.close();
	const conversation = await b.writer.getConversation(conversationId);
	if (!conversation) throw new Error('The conversation is missing.');
	return {
		records: await harnessRecords(streams, pathOf(agentName)),
		context: buildConversationContext(conversation).map((message) => [
			message.role,
			typeof message.content === 'string'
				? message.content
				: message.content
						.map((block) =>
							block.type === 'text' ? block.text : block.type === 'toolCall' ? 'call' : '',
						)
						.join(''),
		]),
		conversation,
	};
}

/** Every harness log record of the instance stream at `path`, in order. */
async function harnessRecords(streams: ConversationStreamStore, path: string) {
	const page = await streams.read(path);
	return page.batches.flatMap((batch) => harnessLogRecordsOf(batch.records));
}

/** Resolves when the harness log at `path` has a record that `match` accepts. */
async function logged(
	streams: ConversationStreamStore,
	path: string,
	match: (record: HarnessLogRecord) => boolean,
) {
	for (;;) {
		if ((await harnessRecords(streams, path)).some(match)) return;
		await new Promise((resolve) => setTimeout(resolve, 20));
	}
}

/** The text of each message that a model call got, with its role. */
function seenMessages(context: FauxContext) {
	return context.messages.map((message) => {
		if (message.role === 'signal') return [message.role, message.content];
		const { content } = message;
		const text =
			typeof content === 'string'
				? content
				: content
						.map((block) =>
							block.type === 'text'
								? block.text
								: block.type === 'toolCall'
									? `call ${block.name}`
									: '',
						)
						.join('');
		return [message.role, text];
	});
}

/** A promise that rejects when `signal` aborts. */
function untilAborted(signal: AbortSignal | undefined) {
	return new Promise<never>((_, reject) => {
		signal?.addEventListener('abort', () => reject(new Error('Aborted.')), { once: true });
	});
}

/** A model call that waits for its host to stop, and then ends as aborted. */
function cutModelCall(called: PromiseWithResolvers<void>, stopped: Promise<void>) {
	return async () => {
		called.resolve();
		await stopped;
		return fauxAssistantMessage([], { stopReason: 'aborted' });
	};
}

describe('harness recovery on the durable path', () => {
	it('closes a cut tool call with the interrupted marker and does not run it again', async () => {
		const seen: string[][][] = [];
		const started = Promise.withResolvers<void>();
		let runs = 0;
		function LookupAgent() {
			useModel('faux/model');
			useTool({
				name: 'lookup',
				description: 'Look the value up.',
				run: async ({ signal }) => {
					runs += 1;
					started.resolve();
					return untilAborted(signal);
				},
			});
			return 'Look the value up.';
		}
		const { records, context } = await stopAndRecover({
			agent: LookupAgent,
			responses: [
				fauxAssistantMessage([fauxToolCall('lookup', {}, { id: 'call_1' })], {
					stopReason: 'toolUse',
				}),
				(request) => {
					seen.push(seenMessages(request));
					return fauxAssistantMessage([fauxText('The lookup was cut.')]);
				},
			],
			stopPoint: () => started.promise,
		});

		expect(runs).toBe(1);
		expect(seen).toEqual([
			[
				['user', 'Go.'],
				['assistant', 'call lookup'],
				['toolResult', INTERRUPTED_TEXT],
			],
		]);
		const repairs = records.filter(
			(record) =>
				record.type === 'tool_outcome' &&
				typeof record.id === 'string' &&
				record.id.startsWith('record_tool_repair_outcome_'),
		);
		expect(repairs.map((record) => [record.toolCallId, record.content])).toEqual([
			['call_1', [{ type: 'text', text: INTERRUPTED_TEXT }]],
		]);
		expect(
			records
				.filter((record) => record.type === 'tool_results_committed')
				.map((record) => String(record.id).startsWith('record_tool_repair_commit_')),
		).toEqual([true]);
		expect(context).toEqual([
			['user', 'Go.'],
			['assistant', 'call'],
			['toolResult', INTERRUPTED_TEXT],
			['assistant', 'The lookup was cut.'],
		]);
	});

	it('runs a cut durable tool call again, and its finished step returns the stored value', async () => {
		const seen: string[][][] = [];
		const stepped = Promise.withResolvers<void>();
		let charges = 0;
		let runs = 0;
		function ChargeAgent() {
			useModel('faux/model');
			useTool({
				name: 'charge',
				description: 'Charge the card.',
				durable: true,
				run: async ({ step, signal }) => {
					runs += 1;
					const receipt = await step.do('charge', async () => {
						charges += 1;
						return `receipt-${charges}`;
					});
					if (runs === 1) {
						stepped.resolve();
						return untilAborted(signal);
					}
					return `charged with ${receipt}`;
				},
			});
			return 'Charge the card.';
		}
		const { records, context } = await stopAndRecover({
			agent: ChargeAgent,
			responses: [
				fauxAssistantMessage([fauxToolCall('charge', {}, { id: 'call_1' })], {
					stopReason: 'toolUse',
				}),
				(request) => {
					seen.push(seenMessages(request));
					return fauxAssistantMessage([fauxText('Charged.')]);
				},
			],
			stopPoint: async (streams) => {
				await stepped.promise;
				await logged(
					streams,
					pathOf('ChargeAgent'),
					(record) => record.type === 'harness.tool.step',
				);
			},
		});

		expect(runs).toBe(2);
		expect(charges).toBe(1);
		expect(seen).toEqual([
			[
				['user', 'Go.'],
				['assistant', 'call charge'],
				['toolResult', '"charged with receipt-1"'],
			],
		]);
		expect(
			records
				.filter((record) => record.type === 'tool_outcome')
				.map((record) => [String(record.id).startsWith('record_tool_outcome_'), record.content]),
		).toEqual([[true, [{ type: 'text', text: '"charged with receipt-1"' }]]]);
		expect(
			records
				.filter((record) => record.type === 'tool_results_committed')
				.map((record) => String(record.id).startsWith('record_tool_repair_commit_')),
		).toEqual([true]);
		expect(context.at(-1)).toEqual(['assistant', 'Charged.']);
	});

	it('reattaches a cut task call to its child conversation', async () => {
		const seen: string[][][] = [];
		const childCalled = Promise.withResolvers<void>();
		const stopped = Promise.withResolvers<void>();
		function Helper() {
			return 'Find what you are asked for.';
		}
		function TaskAgent() {
			useModel('faux/model');
			useSubagent({ name: 'helper', description: 'Finds things.', agent: Helper });
			return 'Delegate the search.';
		}
		const { records, context } = await stopAndRecover({
			agent: TaskAgent,
			responses: [
				fauxAssistantMessage(
					[fauxToolCall('task', { agent: 'helper', prompt: 'Find the key.' }, { id: 'call_1' })],
					{ stopReason: 'toolUse' },
				),
				cutModelCall(childCalled, stopped.promise),
				fauxAssistantMessage([fauxText('The key is under the mat.')]),
				(request) => {
					seen.push(seenMessages(request));
					return fauxAssistantMessage([fauxText('It is under the mat.')]);
				},
			],
			stopPoint: () => childCalled.promise,
			onStop: () => stopped.resolve(),
		});

		expect(records.filter((record) => record.type === 'child_session_retained')).toHaveLength(1);
		expect(seen).toEqual([
			[
				['user', 'Go.'],
				['assistant', 'call task'],
				['toolResult', 'The key is under the mat.'],
			],
		]);
		expect(context.at(-1)).toEqual(['assistant', 'It is under the mat.']);
	});
});

describe('a cut answer on the durable path', () => {
	const answer = `${'The answer goes on. '.repeat(100)}End.`;

	it('keeps the streamed text and continues it after the two recovery signals', async () => {
		const seen: string[][][] = [];
		function StreamAgent() {
			useModel('faux/model');
			return 'Answer at length.';
		}
		const path = pathOf('StreamAgent');
		const { records, context, conversation } = await stopAndRecover({
			agent: StreamAgent,
			responses: [
				fauxAssistantMessage([fauxText(answer)]),
				(request) => {
					seen.push(seenMessages(request));
					return fauxAssistantMessage([fauxText('Done.')]);
				},
			],
			stopPoint: async (streams) => {
				await logged(streams, path, (record) => record.type === 'assistant_text_delta');
				await logged(
					streams,
					path,
					(record) =>
						record.type === 'harness.event' &&
						typeof record.event === 'object' &&
						record.event !== null &&
						'type' in record.event &&
						record.event.type === 'TEXT_MESSAGE_CONTENT',
				);
			},
			tokensPerSecond: 100,
		});

		const partial = [...conversation.entries.values()].find(
			(entry) =>
				entry.type === 'message' &&
				entry.message.role === 'assistant' &&
				entry.message.stopReason === 'aborted',
		);
		if (partial?.type !== 'message' || partial.message.role !== 'assistant')
			throw new Error('The aborted partial is missing.');
		const partialText = partial.message.content
			.map((block) => (block.type === 'text' ? block.text : ''))
			.join('');
		expect(partialText.length).toBeGreaterThan(0);
		expect(answer.startsWith(partialText)).toBe(true);
		expect(partial.message.errorMessage).toBe('Stream interrupted before completion.');
		expect(
			records
				.filter((record) => String(record.id).startsWith(`record_recovery_${partial.id}`))
				.map((record) => record.id),
		).toEqual([
			...[...partial.message.content.keys()].map(() => expect.stringMatching(/_completed$/)),
			`record_recovery_${partial.id}_aborted`,
			`record_recovery_${partial.id}_stream_interrupted`,
			`record_recovery_${partial.id}_stream_continued`,
		]);

		expect(seen).toHaveLength(1);
		const [user, assistant, ...notes] = seen[0] ?? [];
		expect(user).toEqual(['user', 'Go.']);
		expect(assistant?.[0]).toBe('assistant');
		expect(assistant?.[1]?.length).toBeGreaterThan(0);
		expect(answer.startsWith(assistant?.[1] ?? 'missing')).toBe(true);
		expect(notes).toEqual([
			['user', INTERRUPTED_NOTE],
			['user', CONTINUED_NOTE],
		]);
		expect(context.slice(2)).toEqual([
			['user', INTERRUPTED_NOTE],
			['user', CONTINUED_NOTE],
			['assistant', 'Done.'],
		]);
	}, 30_000);

	it('runs a cut model call again with no note when it streamed no text', async () => {
		const seen: string[][][] = [];
		const called = Promise.withResolvers<void>();
		const stopped = Promise.withResolvers<void>();
		function QuietAgent() {
			useModel('faux/model');
			return 'Answer briefly.';
		}
		const { records, context } = await stopAndRecover({
			agent: QuietAgent,
			responses: [
				cutModelCall(called, stopped.promise),
				(request) => {
					seen.push(seenMessages(request));
					return fauxAssistantMessage([fauxText('Brief.')]);
				},
			],
			stopPoint: () => called.promise,
			onStop: () => stopped.resolve(),
		});

		expect(seen).toEqual([[['user', 'Go.']]]);
		expect(records.filter((record) => record.type === 'signal')).toEqual([]);
		expect(context).toEqual([
			['user', 'Go.'],
			['assistant', 'Brief.'],
		]);
	});
});
