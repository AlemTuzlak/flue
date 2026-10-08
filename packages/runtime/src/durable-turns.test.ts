import { logMessageStore } from '@tanstack/ai-harness';
import * as v from 'valibot';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { AgentLoop } from './agent-loop.ts';
import { createFlueContext } from './client.ts';
import type { HarnessLogRecord } from './conversation-records.ts';
import { buildConversationContext } from './conversation-reducer.ts';
import { ConversationRecordWriter } from './conversation-writer.ts';
import { useAgentFinish, useModel, useTool } from './index.ts';
import { sqlite } from './node/agent-execution-store.ts';
import { setProvider } from './providers/registry.ts';
import { buildResultFollowUpPrompt } from './result.ts';
import {
	createAgentSubmissionSessionHandler,
	createDirectAgentSubmissionInput,
	ensureInstanceIdentity,
	type SubmissionJoinSource,
} from './runtime/agent-submissions.ts';
import { InMemoryAttachmentStore } from './runtime/attachment-store.ts';
import type { ConversationStreamStore } from './runtime/conversation-stream-store.ts';
import { createFlueLogStore } from './runtime/harness-log-store.ts';
import {
	createHostRecordAppend,
	createInstanceHarnessHost,
	projectFlueRecord,
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

function identityOf(agentName: string) {
	return { agentName, instanceId };
}

function pathOf(agentName: string) {
	return `agents/${agentName}/${instanceId}`;
}

async function openSqliteStores() {
	const adapter = sqlite();
	await adapter.migrate?.();
	const stores = await adapter.connect();
	return { submissions: stores.submissionStore, streams: stores.conversationStreamStore };
}

/** A Flue context whose sessions run on an instance host over new SQLite stores. */
async function durableContext(agentName: string, responses: FauxResponseStep[]) {
	const faux = fauxProvider({ models: [{ id: 'model' }] });
	faux.setResponses(responses);
	setProvider(faux.provider);
	const identity = identityOf(agentName);
	const path = pathOf(agentName);
	const { streams, submissions } = await openSqliteStores();
	const binding = {
		host: createInstanceHarnessHost({ streams, submissions, path, identity, ownerId: 'host-a' }),
		logId: path,
	};
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
	return { ctx, faux, binding, streams, submissions, writer, path };
}

/** The text of each message that a model call got, with its role. Images read as `[image]`. */
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
									: block.type === 'image'
										? '[image]'
										: '',
						)
						.join('');
		return [message.role, text];
	});
}

/** The transcript of `threadId` in the instance stream at `path`, read by a new reader. */
async function storedTranscript(streams: ConversationStreamStore, path: string, threadId: string) {
	const log = createFlueLogStore(streams, {
		producerId: 'reader',
		identityFor: () => ({ agentName: path.split('/')[1] ?? '', instanceId }),
	});
	const messages = await logMessageStore({
		store: log,
		logId: path,
		project: { record: projectFlueRecord, version: 'flue-signals-1' },
	}).loadThread(threadId);
	return messages.map(({ role, content, toolCalls }) => [
		role,
		typeof content === 'string' || content === null
			? (content ?? toolCalls?.map((call) => `call ${call.function.name}`).join('') ?? '')
			: content.map((part) => (part.type === 'text' ? part.content : `[${part.type}]`)).join(''),
	]);
}

/** Every harness log record of the instance stream at `path`, in order. */
async function harnessRecords(streams: ConversationStreamStore, path: string) {
	const page = await streams.read(path);
	const records: HarnessLogRecord[] = [];
	for (const batch of page.batches) {
		for (const record of batch.records) {
			if (record.type === 'harness_log_batch') records.push(...record.records);
		}
	}
	return records;
}

/** A direct submission of `agentName`, admitted and claimed by `attempt-1` of `host-a`. */
async function claimedSubmission(
	submissions: Awaited<ReturnType<typeof openSqliteStores>>['submissions'],
	agentName: string,
	body: string,
) {
	const input = await createDirectAgentSubmissionInput({
		agent: agentName,
		id: instanceId,
		message: { kind: 'user', body },
	});
	await submissions.admitDirect(input);
	await submissions.markSubmissionCanonicalReady(input.submissionId);
	const attempt = { submissionId: input.submissionId, attemptId: 'attempt-1' };
	await submissions.claimSubmission({
		...attempt,
		ownerId: 'host-a',
		leaseExpiresAt: Date.now() + 30_000,
	});
	return { input, attempt };
}

describe('transient model retries on the durable path', () => {
	const identity = identityOf('RetryAgent');
	const path = pathOf('RetryAgent');
	const threadId = 'conversation-1';
	const overloaded = () =>
		fauxAssistantMessage([], { stopReason: 'error', errorMessage: '503 service unavailable' });

	afterEach(() => {
		vi.useRealTimers();
	});

	/** A loop of `host` on the thread. `recover` gives a recovered turn the faux adapter. */
	function loopOn(
		host: ReturnType<typeof createInstanceHarnessHost>,
		faux: ReturnType<typeof fauxProvider>,
	) {
		const { createAdapter } = faux.provider;
		const model = faux.getModel();
		if (!createAdapter || !model) throw new Error('The faux provider has no adapter or model.');
		const adapterFor = () => createAdapter(model, { auth: undefined, promptCacheKey: threadId });
		return new AgentLoop({
			initialState: { systemPrompt: 'Be brief.', model, tools: [], thinkingLevel: 'off' },
			sessionId: threadId,
			createAdapter: async () => adapterFor(),
			durable: { host, threadId, logId: path },
			recover: async () => ({ action: 'run', overrides: { adapter: await adapterFor() } }),
		});
	}

	/** The `retries` of each retry record of the log, in order. */
	async function retryCounts(streams: ConversationStreamStore) {
		return (await harnessRecords(streams, path))
			.filter((record) => record.type === 'harness.turn.retry')
			.map((record) => record.retries);
	}

	/**
	 * Host A gets two retryable errors and stops in the backoff after the
	 * second. Host B opens the thread, and the turn goes on there with
	 * `afterRestart` as the model's answers.
	 */
	async function retryAcrossRestart(afterRestart: FauxResponseStep[]) {
		vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'], shouldAdvanceTime: true });
		const { streams } = await openSqliteStores();
		const faux = fauxProvider();
		faux.setResponses([overloaded(), overloaded(), ...afterRestart]);
		const hostA = createInstanceHarnessHost({ streams, path, identity, ownerId: 'host-a' });
		const loopA = loopOn(hostA, faux);
		const runA = loopA.prompt('Look it up.', undefined, { inputId: 'submission-1' });
		await vi.waitFor(async () => expect(await retryCounts(streams)).toEqual([1, 2]), {
			timeout: 10_000,
		});
		await hostA.close({ recoverable: true });
		await runA;

		const hostB = createInstanceHarnessHost({ streams, path, identity, ownerId: 'host-b' });
		const loopB = loopOn(hostB, faux);
		const messages: string[] = [];
		loopB.subscribe((event) => {
			if (event.type === 'message_end' && event.message.role === 'assistant')
				messages.push(event.message.errorMessage ?? summarizeText(event.message.content));
		});
		let settled = false;
		// The same input id and message: the loop follows the turn that recovery runs.
		const runB = loopB.prompt('Look it up.', undefined, { inputId: 'submission-1' }).finally(() => {
			settled = true;
		});
		while (!settled) await vi.advanceTimersByTimeAsync(1_000);
		await runB;
		const counts = await retryCounts(streams);
		await hostB.close();
		return { faux, loopB, messages, counts, streams };
	}

	it('retries a submission inside the harness turn only, and fails it at the fourth error', async () => {
		vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'], shouldAdvanceTime: true });
		function BriefAgent() {
			useModel('faux/model');
			return 'Be brief.';
		}
		const { ctx, faux, submissions, writer, binding } = await durableContext('BriefAgent', [
			overloaded(),
			overloaded(),
			overloaded(),
			overloaded(),
			fauxAssistantMessage([fauxText('Too late.')]),
		]);
		await ensureInstanceIdentity(writer, BriefAgent, undefined);
		const { input, attempt } = await claimedSubmission(submissions, 'BriefAgent', 'Hi.');

		let outcome: unknown;
		const run = createAgentSubmissionSessionHandler(BriefAgent, input, (session) =>
			session.processSubmissionInput(input, { submissionAttempt: attempt }),
		)(ctx).then(
			() => (outcome = 'completed'),
			(error: unknown) => (outcome = error),
		);
		while (outcome === undefined) await vi.advanceTimersByTimeAsync(1_000);
		await run;

		expect(outcome).toBeInstanceOf(Error);
		expect(outcome instanceof Error ? outcome.message : outcome).toContain(
			'503 service unavailable',
		);
		await binding.host.close();
		expect(faux.state.callCount).toBe(4);
	});

	it('fails the turn at the fourth error, counting the retries from before the restart', async () => {
		const { faux, loopB, messages, counts } = await retryAcrossRestart([
			overloaded(),
			overloaded(),
		]);

		expect(faux.state.callCount).toBe(4);
		expect(counts).toEqual([1, 2, 3, 4]);
		expect(messages).toEqual(['503 service unavailable', '503 service unavailable']);
		expect(loopB.state.errorMessage).toBe('503 service unavailable');
	});

	it('completes the turn with an answer after the third retry', async () => {
		const { faux, loopB, messages, counts, streams } = await retryAcrossRestart([
			overloaded(),
			fauxAssistantMessage([fauxText('Found it.')]),
		]);

		expect(faux.state.callCount).toBe(4);
		expect(counts).toEqual([1, 2, 3]);
		expect(messages).toEqual(['503 service unavailable', 'Found it.']);
		expect(loopB.state.errorMessage).toBeUndefined();
		expect(await storedTranscript(streams, path, threadId)).toEqual([
			['user', 'Look it up.'],
			['assistant', 'Found it.'],
		]);
	});
}, 60_000);

function summarizeText(content: readonly { type: string; text?: string }[]) {
	return content.map((block) => block.text ?? '').join('');
}

describe('a join on the durable path', () => {
	it('joins a delivery that arrives during a tool call into the running turn', async () => {
		const seen: string[][][] = [];
		const second = await createDirectAgentSubmissionInput({
			agent: 'JoinAgent',
			id: instanceId,
			message: { kind: 'user', body: 'And the boxes?' },
		});
		let admitSecond: () => Promise<void> = async () => {};
		function JoinAgent() {
			useModel('faux/model');
			useTool({
				name: 'count',
				description: 'Count the items.',
				run: async () => {
					await admitSecond();
					return 'counted';
				},
			});
			return 'Count the items, then answer.';
		}
		const { ctx, streams, submissions, writer, path, binding } = await durableContext('JoinAgent', [
			fauxAssistantMessage([fauxToolCall('count', {}, { id: 'call_1' })], {
				stopReason: 'toolUse',
			}),
			(context) => {
				seen.push(seenMessages(context));
				return fauxAssistantMessage([fauxText('3 items and 2 boxes.')]);
			},
		]);
		admitSecond = async () => {
			await submissions.admitDirect(second);
			await submissions.markSubmissionCanonicalReady(second.submissionId);
		};
		const { conversationId } = await ensureInstanceIdentity(writer, JoinAgent, undefined);
		const first = await createDirectAgentSubmissionInput({
			agent: 'JoinAgent',
			id: instanceId,
			message: { kind: 'user', body: 'How many items?' },
		});
		await submissions.admitDirect(first);
		await submissions.markSubmissionCanonicalReady(first.submissionId);
		await submissions.claimSubmission({
			submissionId: first.submissionId,
			attemptId: 'attempt-1',
			ownerId: 'host-a',
			leaseExpiresAt: Date.now() + 30_000,
		});
		const attempt = { submissionId: first.submissionId, attemptId: 'attempt-1' };
		const joinSource: SubmissionJoinSource = {
			claim: () => submissions.claimJoinableSubmissions(attempt, 'JoinAgent'),
			finalize: (submissionId) => submissions.finalizeJoinedSubmission(attempt, submissionId),
			revert: (submissionId) => submissions.revertJoiningSubmission(attempt, submissionId),
			listUnresolved: () => submissions.listJoinedSubmissions(attempt.submissionId),
		};

		await createAgentSubmissionSessionHandler(JoinAgent, first, (session) =>
			session.processSubmissionInput(first, { submissionAttempt: attempt, joinSource }),
		)(ctx);
		await binding.host.close();

		expect(seen).toEqual([
			[
				['user', 'How many items?'],
				['assistant', 'call count'],
				['toolResult', '"counted"'],
				['user', 'And the boxes?'],
			],
		]);
		expect((await submissions.getSubmission(second.submissionId))?.status).toBe('joined');
		const records = await harnessRecords(streams, path);
		expect(
			records
				.filter((record) => record.type === 'harness.input.joined')
				.map((record) => [record.inputId, record.into]),
		).toEqual([[second.submissionId, first.submissionId]]);
		expect(
			records
				.filter((record) => record.type === 'harness.input.settled')
				.map((record) => [record.inputId, record.outcome]),
		).toEqual([
			[first.submissionId, 'completed'],
			[second.submissionId, 'completed'],
		]);
		// The input record of the joined delivery lands in the append that joins it.
		const joinBatch = (await streams.read(path)).batches
			.flatMap((batch) => batch.records)
			.filter((record) => record.type === 'harness_log_batch')
			.map((record) => record.records)
			.find((batch) => batch.some((record) => record.type === 'harness.input.joined'));
		expect(
			joinBatch
				?.filter((record) => !record.type.startsWith('harness.'))
				.map((record) => [record.type, record.messageId]),
		).toEqual([
			['user_message', `entry_direct_${Buffer.from(second.submissionId).toString('base64url')}`],
		]);
		expect(await storedTranscript(streams, path, conversationId)).toEqual([
			['user', 'How many items?'],
			['assistant', 'call count'],
			['tool', '"counted"'],
			['user', 'And the boxes?'],
			['assistant', '3 items and 2 boxes.'],
		]);
	});
});

describe('useAgentFinish on the durable path', () => {
	it('continues the same harness turn with the appended signal', async () => {
		const seen: string[][][] = [];
		let appended = false;
		function FinishAgent() {
			useModel('faux/model');
			useAgentFinish(({ append }) => {
				if (appended) return;
				appended = true;
				append({ kind: 'signal', type: 'reminder', body: 'Say bye.' });
			});
			return 'Answer the user.';
		}
		const { ctx, faux, streams, submissions, writer, path, binding } = await durableContext(
			'FinishAgent',
			[
				fauxAssistantMessage([fauxText('Hello.')]),
				(context) => {
					seen.push(seenMessages(context));
					return fauxAssistantMessage([fauxText('Bye.')]);
				},
			],
		);
		const { conversationId } = await ensureInstanceIdentity(writer, FinishAgent, undefined);
		const input = await createDirectAgentSubmissionInput({
			agent: 'FinishAgent',
			id: instanceId,
			message: { kind: 'user', body: 'Hi.' },
		});
		await submissions.admitDirect(input);
		await submissions.markSubmissionCanonicalReady(input.submissionId);
		const attempt = { submissionId: input.submissionId, attemptId: 'attempt-1' };
		await submissions.claimSubmission({
			...attempt,
			ownerId: 'host-a',
			leaseExpiresAt: Date.now() + 30_000,
		});

		await createAgentSubmissionSessionHandler(FinishAgent, input, (session) =>
			session.processSubmissionInput(input, { submissionAttempt: attempt }),
		)(ctx);
		await binding.host.close();

		expect(faux.state.callCount).toBe(2);
		expect(seen).toEqual([
			[
				['user', 'Hi.'],
				['assistant', 'Hello.'],
				['user', '<signal type="reminder">\nSay bye.\n</signal>'],
			],
		]);
		const records = await harnessRecords(streams, path);
		expect(
			records
				.filter((record) => record.type === 'harness.input.settled')
				.map((record) => [record.inputId, record.outcome]),
		).toEqual([[input.submissionId, 'completed']]);
		// The signal and the cycle record land in one append.
		const cycleBatch = (await streams.read(path)).batches
			.flatMap((batch) => batch.records)
			.filter((record) => record.type === 'harness_log_batch')
			.map((record) => record.records)
			.find((batch) => batch.some((record) => record.type === 'agent_finish_cycle'));
		expect(cycleBatch?.map((record) => record.type)).toEqual(['signal', 'agent_finish_cycle']);
		// The faux adapter names each answer after its run, and the harness runs
		// a continued turn with the same run id, so the transcript keeps the
		// first answer of the turn only. Flue's history has both.
		const conversation = await writer.getConversation(conversationId);
		if (!conversation) throw new Error('The conversation is missing.');
		expect(
			buildConversationContext(conversation).map((message) => [
				message.role,
				typeof message.content === 'string'
					? message.content
					: message.content.map((block) => (block.type === 'text' ? block.text : '')).join(''),
			]),
		).toEqual([
			['user', 'Hi.'],
			['assistant', 'Hello.'],
			['user', '<signal type="reminder">\nSay bye.\n</signal>'],
			['assistant', 'Bye.'],
		]);
	});
});

describe('the structured-result reminder on the durable path', () => {
	function CountAgent() {
		useModel('faux/model');
		return 'Count the items.';
	}

	it('reaches each later model call of the turn and stays out of the transcript', async () => {
		const seen: string[][][] = [];
		const reminder = buildResultFollowUpPrompt();
		const { ctx, streams, path, binding, writer } = await durableContext('CountAgent', [
			fauxAssistantMessage([fauxText('There are 3.')]),
			(context) => {
				seen.push(seenMessages(context));
				return fauxAssistantMessage(
					[fauxToolCall('finish', { count: 'three' }, { id: 'call_1' })],
					{
						stopReason: 'toolUse',
					},
				);
			},
			(context) => {
				seen.push(seenMessages(context));
				return fauxAssistantMessage([fauxToolCall('finish', { count: 3 }, { id: 'call_2' })], {
					stopReason: 'toolUse',
				});
			},
		]);
		const { conversationId } = await ensureInstanceIdentity(writer, CountAgent, undefined);
		const harness = await ctx.initializeRootHarness(CountAgent);

		const response = await harness.prompt('How many items?', {
			result: v.object({ count: v.number() }),
		});
		await binding.host.close();

		expect(response.data).toEqual({ count: 3 });
		expect(seen.map((messages) => messages.map(([role]) => role))).toEqual([
			['user', 'assistant', 'user'],
			['user', 'assistant', 'user', 'assistant', 'toolResult'],
		]);
		expect(seen.map((messages) => [messages[1]?.[1], messages[2]?.[1]])).toEqual([
			['There are 3.', reminder],
			['There are 3.', reminder],
		]);
		const transcript = await storedTranscript(streams, path, conversationId);
		expect(transcript.some(([, text]) => text === reminder)).toBe(false);
		expect(transcript.map(([role]) => role)).toEqual([
			'user',
			'assistant',
			'assistant',
			'tool',
			'assistant',
			'tool',
		]);
	});
});

describe('a compaction on the durable path', () => {
	function BriefAgent() {
		useModel('faux/model', { compaction: { keepRecentTokens: 1 } });
		return 'Be brief.';
	}

	it('gives the next model call the compacted context and stores it as the transcript', async () => {
		const seen: string[][][] = [];
		const { ctx, streams, path, binding, writer } = await durableContext('BriefAgent', [
			fauxAssistantMessage([fauxText('One.')]),
			fauxAssistantMessage([fauxText('Two.')]),
			// The summary, then the summary of the kept turn's prefix.
			fauxAssistantMessage([fauxText('They said hi.')]),
			fauxAssistantMessage([fauxText('They said hi again.')]),
			(context) => {
				seen.push(seenMessages(context));
				return fauxAssistantMessage([fauxText('Three.')]);
			},
		]);
		const { conversationId } = await ensureInstanceIdentity(writer, BriefAgent, undefined);
		const harness = await ctx.initializeRootHarness(BriefAgent);
		await harness.prompt('Hi.');
		await harness.prompt('Hi again.');
		await harness.compact();
		await harness.prompt('Bye.');
		await binding.host.close();

		const compacted = [
			[
				'user',
				'<compaction type="context_summary">\nThey said hi.\n\n---\n\n**Turn Context (split turn):**\n\nThey said hi again.\n</compaction>',
			],
			['assistant', 'Two.'],
			['user', 'Bye.'],
		];
		expect(seen).toEqual([compacted]);
		expect(await storedTranscript(streams, path, conversationId)).toEqual([
			...compacted,
			['assistant', 'Three.'],
		]);
	});
});
