import { afterEach, describe, expect, it, vi } from 'vitest';
import type { AgentSubmissionStore } from './agent-execution-store.ts';
import { createFlueContext } from './client.ts';
import type { HarnessLogRecord } from './conversation-records.ts';
import { ConversationRecordWriter } from './conversation-writer.ts';
import { type AgentFunction, useModel, useTool } from './index.ts';
import { sqlite } from './node/agent-execution-store.ts';
import { setProvider } from './providers/registry.ts';
import {
	type AgentSubmissionInput,
	createAgentSubmissionSessionHandler,
	createDirectAgentSubmissionInput,
	ensureInstanceIdentity,
	processSubmission,
	reconcileInterruptedSubmission,
} from './runtime/agent-submissions.ts';
import { InMemoryAttachmentStore } from './runtime/attachment-store.ts';
import type { ConversationStreamStore } from './runtime/conversation-stream-store.ts';
import { generateAttemptId, generateSubmissionId } from './runtime/ids.ts';
import {
	createHostRecordAppend,
	createInstanceHarnessHost,
} from './runtime/instance-harness-host.ts';
import { resolveModel } from './runtime/providers.ts';
import { registerFlueAgents, resetFlueAgentRegistrationForTests } from './runtime/registration.ts';
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
	return { ctx, host, writer, ownerId };
}

type Host = Awaited<ReturnType<typeof hostContext>>;

/** Admit a direct submission with `body`, ready to claim. */
async function admit(submissions: AgentSubmissionStore, agent: AgentFunction, body: string) {
	const input = await createDirectAgentSubmissionInput({
		agent: agent.name,
		id: instanceId,
		message: { kind: 'user', body },
	});
	await submissions.admitDirect(input);
	await submissions.markSubmissionCanonicalReady(input.submissionId);
	return input;
}

/** Claim the submission for `host`, as a coordinator does. */
async function claim(submissions: AgentSubmissionStore, input: AgentSubmissionInput, host: Host) {
	const claimed = await submissions.claimSubmission({
		submissionId: input.submissionId,
		attemptId: generateAttemptId(),
		ownerId: host.ownerId,
		leaseExpiresAt: Date.now() + LEASE_MS,
	});
	if (!claimed) throw new Error('The submission was not claimed.');
	return claimed;
}

/** Run a claimed submission on `host`. A stop leaves its row running, as a shutdown does. */
async function run(
	submissions: AgentSubmissionStore,
	agent: AgentFunction,
	submissionId: string,
	host: Host,
) {
	const submission = await submissions.getSubmission(submissionId);
	if (!submission) throw new Error('The submission is missing.');
	await processSubmission({
		submissions,
		submission,
		resolveAgent: () => agent,
		createContext: () => host.ctx,
		conversationWriter: host.writer,
		isShutdownAbort: () => true,
	});
}

/** Reconcile the interrupted submission on `host`, and run its replacement attempt there. */
async function recover(
	submissions: AgentSubmissionStore,
	agent: AgentFunction,
	submissionId: string,
	host: Host,
) {
	const interrupted = await submissions.getSubmission(submissionId);
	if (!interrupted) throw new Error('The submission is missing.');
	const replacement = await reconcileInterruptedSubmission(
		submissions,
		interrupted,
		agent,
		() => host.ctx,
		{ ownerId: host.ownerId, leaseExpiresAt: Date.now() + LEASE_MS },
		host.writer,
	);
	if (!replacement) throw new Error('Reconciliation did not hand out a replacement attempt.');
	await run(submissions, agent, submissionId, host);
}

/**
 * A crash on host A: run the first submission of `agent` there until
 * `stopPoint` resolves, then stop host A with `close({ recoverable: true })`.
 */
async function crashOnHostA(options: {
	agent: AgentFunction;
	responses: FauxResponseStep[];
	stopPoint: () => Promise<void>;
	onStop?: () => void;
	/** Runs before host A starts the submission. */
	beforeRun?: (stores: Awaited<ReturnType<typeof openStores>>) => void;
}) {
	const { agent } = options;
	const faux = fauxProvider({ models: [{ id: 'model' }] });
	faux.setResponses(options.responses);
	setProvider(faux.provider);
	const stores = await openStores();
	options.beforeRun?.(stores);
	const { streams, submissions } = stores;
	const a = await hostContext(agent.name, streams, 'host-a');
	await ensureInstanceIdentity(a.writer, agent, undefined);
	const input = await admit(submissions, agent, 'Go.');
	await claim(submissions, input, a);
	const runA = run(submissions, agent, input.submissionId, a).catch(() => undefined);
	await options.stopPoint();
	options.onStop?.();
	await a.host.close({ recoverable: true });
	await runA;
	return { streams, submissions, input };
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

/** The outcome of each `harness.input.settled` record of `inputId`. */
async function harnessSettlements(streams: ConversationStreamStore, path: string, inputId: string) {
	return (await harnessRecords(streams, path))
		.filter((record) => record.type === 'harness.input.settled' && record.inputId === inputId)
		.map((record) => record.outcome);
}

/** The record `id` of the instance stream at `path`. */
async function logRecord(streams: ConversationStreamStore, path: string, id: string) {
	return (await harnessRecords(streams, path)).find((record) => record.id === id);
}

/** The `submission_settled` record of a direct submission. */
async function settledRecord(streams: ConversationStreamStore, path: string, submissionId: string) {
	return logRecord(streams, path, `record_direct-submission:${submissionId}:settled`);
}

/** The outcome of the `submission_settled` record of a direct submission. */
async function settledOutcome(
	streams: ConversationStreamStore,
	path: string,
	submissionId: string,
) {
	return (await settledRecord(streams, path, submissionId))?.outcome;
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

/**
 * Open the instance on host C and run one more submission. The model call
 * it makes is the only one: the settled input does not run again.
 */
async function runNextSubmission(
	streams: ConversationStreamStore,
	submissions: AgentSubmissionStore,
	agent: AgentFunction,
	calls: () => number,
) {
	const before = calls();
	const c = await hostContext(agent.name, streams, 'host-c');
	const next = await admit(submissions, agent, 'Next.');
	await claim(submissions, next, c);
	await run(submissions, agent, next.submissionId, c);
	await c.host.close();
	return {
		calls: calls() - before,
		status: (await submissions.getSubmission(next.submissionId))?.status,
		outcome: await settledOutcome(streams, pathOf(agent.name), next.submissionId),
	};
}

afterEach(() => {
	vi.restoreAllMocks();
	resetFlueAgentRegistrationForTests();
});

describe('a recovered submission settles by the ledger, through the harness', () => {
	it('answers a joined delivery after a crash before the next model call', async () => {
		const seen: string[][][] = [];
		const called = Promise.withResolvers<void>();
		const stopped = Promise.withResolvers<void>();
		let submissions: AgentSubmissionStore | undefined;
		const joinedId = generateSubmissionId();
		function JoinAgent() {
			useModel('faux/model');
			useTool({
				name: 'wait',
				description: 'Wait for a moment.',
				run: async () => {
					// A dispatch arrives while the response runs.
					await submissions?.admitDispatch({
						submissionId: joinedId,
						agent: 'JoinAgent',
						id: instanceId,
						message: { kind: 'user', body: 'Also this.' },
						acceptedAt: new Date().toISOString(),
					});
					await submissions?.markSubmissionCanonicalReady(joinedId);
					return 'waited';
				},
			});
			return 'Wait, then answer.';
		}
		const crashed = await crashOnHostA({
			agent: JoinAgent,
			beforeRun: (stores) => {
				submissions = stores.submissions;
			},
			responses: [
				fauxAssistantMessage([fauxToolCall('wait', {}, { id: 'call_1' })], {
					stopReason: 'toolUse',
				}),
				cutModelCall(called, stopped.promise),
				(request) => {
					seen.push(seenMessages(request));
					return fauxAssistantMessage([fauxText('Both are done.')]);
				},
			],
			stopPoint: () => called.promise,
			onStop: () => stopped.resolve(),
		});
		const { streams, input } = crashed;
		const path = pathOf('JoinAgent');
		expect((await crashed.submissions.getSubmission(joinedId))?.status).toBe('joined');

		const b = await hostContext('JoinAgent', streams, 'host-b');
		await recover(crashed.submissions, JoinAgent, input.submissionId, b);
		await b.host.close();

		expect(seen).toEqual([
			[
				['user', 'Go.'],
				['assistant', 'call wait'],
				['toolResult', '"waited"'],
				['user', 'Also this.'],
			],
		]);
		expect((await crashed.submissions.getSubmission(input.submissionId))?.status).toBe('settled');
		expect((await crashed.submissions.getSubmission(joinedId))?.status).toBe('settled');
		const records = await harnessRecords(streams, path);
		const answerAt = records.findLastIndex(
			(record) => record.type === 'assistant_message_completed',
		);
		const settledAt = (submissionId: string) =>
			records.findIndex(
				(record) => record.type === 'submission_settled' && record.submissionId === submissionId,
			);
		expect(answerAt).toBeGreaterThan(-1);
		expect(settledAt(joinedId)).toBeGreaterThan(answerAt);
		expect(settledAt(input.submissionId)).toBeGreaterThan(answerAt);
		expect(
			records
				.filter((record) => record.type === 'submission_settled')
				.map((record) => [record.submissionId, record.outcome]),
		).toEqual([
			[joinedId, 'completed'],
			[input.submissionId, 'completed'],
		]);
		expect(await harnessSettlements(streams, path, input.submissionId)).toEqual(['completed']);
		expect(await harnessSettlements(streams, path, joinedId)).toEqual(['completed']);
	});

	it('settles an abort requested while the host was stopped, with no model call', async () => {
		const called = Promise.withResolvers<void>();
		const stopped = Promise.withResolvers<void>();
		let calls = 0;
		function AbortAgent() {
			useModel('faux/model');
			return 'Answer briefly.';
		}
		const { streams, submissions, input } = await crashOnHostA({
			agent: AbortAgent,
			responses: [
				cutModelCall(called, stopped.promise),
				() => {
					calls += 1;
					return fauxAssistantMessage([fauxText('Next answer.')]);
				},
			],
			stopPoint: () => called.promise,
			onStop: () => stopped.resolve(),
		});
		const path = pathOf('AbortAgent');
		const row = await submissions.getSubmission(input.submissionId);
		if (!row) throw new Error('The submission is missing.');
		await submissions.requestSessionAbort(row.sessionKey);

		const b = await hostContext('AbortAgent', streams, 'host-b');
		await recover(submissions, AbortAgent, input.submissionId, b);
		await b.host.close();

		expect(calls).toBe(0);
		expect((await submissions.getSubmission(input.submissionId))?.status).toBe('settled');
		const settled = await settledRecord(streams, path, input.submissionId);
		expect(settled?.outcome).toBe('aborted');
		expect(settled?.error).toMatchObject({
			type: 'submission_aborted',
			message: 'Submission was aborted.',
		});
		const advisory = await logRecord(
			streams,
			path,
			`record_submission_aborted_${input.submissionId}`,
		);
		expect([advisory?.signalType, advisory?.content]).toEqual([
			'submission_aborted',
			'Submission was aborted.',
		]);
		expect(await harnessSettlements(streams, path, input.submissionId)).toEqual(['aborted']);

		expect(await runNextSubmission(streams, submissions, AbortAgent, () => calls)).toEqual({
			calls: 1,
			status: 'settled',
			outcome: 'completed',
		});
		expect(await harnessSettlements(streams, path, input.submissionId)).toEqual(['aborted']);
	});

	it('settles a finished answer as completed when an abort was requested', async () => {
		let calls = 0;
		function FinishedAgent() {
			useModel('faux/model');
			return 'Answer briefly.';
		}
		const faux = fauxProvider({ models: [{ id: 'model' }] });
		faux.setResponses([
			() => {
				calls += 1;
				return fauxAssistantMessage([fauxText('Done.')]);
			},
			() => {
				calls += 1;
				return fauxAssistantMessage([fauxText('Next answer.')]);
			},
		]);
		setProvider(faux.provider);
		const { streams, submissions } = await openStores();
		const path = pathOf('FinishedAgent');
		const a = await hostContext('FinishedAgent', streams, 'host-a');
		await ensureInstanceIdentity(a.writer, FinishedAgent, undefined);
		const input = await admit(submissions, FinishedAgent, 'Go.');
		const claimed = await claim(submissions, input, a);
		if (!claimed.attemptId) throw new Error('The claim has no attempt.');
		const attempt = { submissionId: input.submissionId, attemptId: claimed.attemptId };
		// Host A answers, and stops before the ledger settles the row.
		await createAgentSubmissionSessionHandler(FinishedAgent, input, (session) =>
			session.processSubmissionInput(input, {
				submissionAttempt: attempt,
				onInputApplied: async (durability) => {
					await submissions.markSubmissionInputApplied(attempt, durability);
				},
			}),
		)(a.ctx);
		await a.host.close({ recoverable: true });
		const row = await submissions.getSubmission(input.submissionId);
		expect(row?.status).toBe('running');
		if (!row) throw new Error('The submission is missing.');
		await submissions.requestSessionAbort(row.sessionKey);

		const b = await hostContext('FinishedAgent', streams, 'host-b');
		await recover(submissions, FinishedAgent, input.submissionId, b);
		await b.host.close();

		expect(calls).toBe(1);
		expect((await submissions.getSubmission(input.submissionId))?.status).toBe('settled');
		const settled = await settledRecord(streams, path, input.submissionId);
		expect(settled?.outcome).toBe('completed');
		expect(await logRecord(streams, path, `record_submission_aborted_${input.submissionId}`)).toBe(
			undefined,
		);
		expect(await harnessSettlements(streams, path, input.submissionId)).toEqual(['completed']);

		expect(await runNextSubmission(streams, submissions, FinishedAgent, () => calls)).toEqual({
			calls: 1,
			status: 'settled',
			outcome: 'completed',
		});
	});

	it('fails a submission past its attempt budget, and names the cut tool call', async () => {
		const started = Promise.withResolvers<void>();
		let runs = 0;
		let calls = 0;
		function BudgetAgent() {
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
		BudgetAgent.durability = { maxAttempts: 1 };
		registerFlueAgents([{ identity: 'BudgetAgent', agent: BudgetAgent }]);
		const { streams, submissions, input } = await crashOnHostA({
			agent: BudgetAgent,
			responses: [
				fauxAssistantMessage([fauxToolCall('lookup', {}, { id: 'call_1' })], {
					stopReason: 'toolUse',
				}),
				() => {
					calls += 1;
					return fauxAssistantMessage([fauxText('Next answer.')]);
				},
			],
			stopPoint: () => started.promise,
		});
		const path = pathOf('BudgetAgent');
		const row = await submissions.getSubmission(input.submissionId);
		expect([row?.attemptCount, row?.maxAttempts]).toEqual([1, 1]);

		const b = await hostContext('BudgetAgent', streams, 'host-b');
		await recover(submissions, BudgetAgent, input.submissionId, b);
		await b.host.close();

		expect([runs, calls]).toEqual([1, 0]);
		expect((await submissions.getSubmission(input.submissionId))?.status).toBe('settled');
		const settled = await settledRecord(streams, path, input.submissionId);
		expect(settled?.outcome).toBe('failed');
		expect(settled?.error).toMatchObject({
			type: 'submission_retry_exhausted',
			message: 'Submission exceeded maximum recovery attempts (1/1).',
			meta: {
				attemptCount: 1,
				maxAttempts: 1,
				interruptedTools: [{ name: 'lookup', id: 'call_1' }],
			},
		});
		const advisory = await logRecord(
			streams,
			path,
			`record_submission_interrupted_${input.submissionId}`,
		);
		expect([advisory?.signalType, advisory?.content]).toEqual([
			'submission_interrupted',
			'Submission exceeded maximum recovery attempts (1/1).\n\nInterrupted tool call(s):\n  - lookup (call_1)',
		]);
		expect(
			(await harnessRecords(streams, path))
				.filter(
					(record) =>
						record.type === 'tool_outcome' &&
						String(record.id).startsWith('record_tool_repair_outcome_'),
				)
				.map((record) => record.toolCallId),
		).toEqual(['call_1']);
		expect(await harnessSettlements(streams, path, input.submissionId)).toEqual(['failed']);

		expect(await runNextSubmission(streams, submissions, BudgetAgent, () => calls)).toEqual({
			calls: 1,
			status: 'settled',
			outcome: 'completed',
		});
		expect(runs).toBe(1);
	});

	it('fails a submission past its deadline', async () => {
		const called = Promise.withResolvers<void>();
		const stopped = Promise.withResolvers<void>();
		let calls = 0;
		function SlowAgent() {
			useModel('faux/model');
			return 'Answer briefly.';
		}
		const { streams, submissions, input } = await crashOnHostA({
			agent: SlowAgent,
			responses: [
				cutModelCall(called, stopped.promise),
				() => {
					calls += 1;
					return fauxAssistantMessage([fauxText('Next answer.')]);
				},
			],
			stopPoint: () => called.promise,
			onStop: () => stopped.resolve(),
		});
		const path = pathOf('SlowAgent');
		// Host B starts two hours later: past the default one hour timeout.
		const realNow = Date.now.bind(Date);
		vi.spyOn(Date, 'now').mockImplementation(() => realNow() + 2 * 3_600_000);
		const row = await submissions.getSubmission(input.submissionId);
		expect((row?.timeoutAt ?? Number.POSITIVE_INFINITY) < Date.now()).toBe(true);

		const b = await hostContext('SlowAgent', streams, 'host-b');
		await recover(submissions, SlowAgent, input.submissionId, b);
		await b.host.close();

		expect(calls).toBe(0);
		expect((await submissions.getSubmission(input.submissionId))?.status).toBe('settled');
		const settled = await settledRecord(streams, path, input.submissionId);
		expect(settled?.outcome).toBe('failed');
		expect(settled?.error).toMatchObject({
			type: 'submission_timeout',
			message: 'Submission exceeded the configured timeout.',
		});
		const advisory = await logRecord(
			streams,
			path,
			`record_submission_interrupted_${input.submissionId}`,
		);
		expect([advisory?.signalType, advisory?.content]).toEqual([
			'submission_interrupted',
			'Submission exceeded the configured timeout.',
		]);
		expect(await harnessSettlements(streams, path, input.submissionId)).toEqual(['failed']);

		expect(await runNextSubmission(streams, submissions, SlowAgent, () => calls)).toEqual({
			calls: 1,
			status: 'settled',
			outcome: 'completed',
		});
	});
});
