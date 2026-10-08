import { expect, it } from 'vitest';
import type { PersistenceAdapter } from './agent-execution-store.ts';
import type { ConversationRecord } from './conversation-records.ts';
import {
	init,
	instrument,
	useAgentStart,
	useModel,
	usePersistentState,
	useSandbox,
} from './index.ts';
import { local, sqlite, start } from './node/index.ts';
import type { ConversationStreamStore } from './runtime/conversation-stream-store.ts';
import { fauxAssistantMessage, fauxProvider, fauxText, fauxToolCall } from './test-utils/faux.ts';

function recordingDatabase() {
	const database = sqlite();
	const records: ConversationRecord[] = [];
	const adapter: PersistenceAdapter = {
		migrate: () => database.migrate?.(),
		close: () => database.close?.(),
		async connect() {
			const stores = await database.connect();
			const stream = stores.conversationStreamStore;
			const recordingStream: ConversationStreamStore = {
				createStream: (...args) => stream.createStream(...args),
				acquireProducer: (...args) => stream.acquireProducer(...args),
				async append(input) {
					const result = await stream.append(input);
					records.push(...structuredClone(input.records));
					return result;
				},
				read: (...args) => stream.read(...args),
				getMeta: (...args) => stream.getMeta(...args),
				subscribe: (...args) => stream.subscribe(...args),
				...(stream.putFoldCheckpoint
					? { putFoldCheckpoint: stream.putFoldCheckpoint.bind(stream) }
					: {}),
				...(stream.getFoldCheckpoint
					? { getFoldCheckpoint: stream.getFoldCheckpoint.bind(stream) }
					: {}),
			};
			return { ...stores, conversationStreamStore: recordingStream };
		},
	};
	return { adapter, records };
}

function waitForAbort(signal: AbortSignal | undefined): Promise<never> {
	if (!signal) throw new Error('Expected the tool to receive an abort signal.');
	return new Promise((_resolve, reject) => {
		if (signal.aborted) reject(signal.reason);
		else
			signal.addEventListener('abort', () => reject(signal.reason), {
				once: true,
			});
	});
}

it('repairs an aborted partial sequential tool batch before recording the failure assistant', async () => {
	const firstStarted = Promise.withResolvers<void>();
	const calls = { first: 0, second: 0 };
	function AbortBatch() {
		useModel('faux/model');
		const [, setPhase] = usePersistentState('phase', 'initial');
		useAgentStart(() => setPhase('started'));
		useSandbox({
			...local(),
			tools: () => [
				{
					name: 'first',
					label: 'First',
					description: 'Wait for the test to abort the session.',
					parameters: { type: 'object', properties: {} },
					executionMode: 'sequential',
					async execute(_id, _args, signal) {
						calls.first += 1;
						setPhase('must-not-commit');
						firstStarted.resolve();
						return await waitForAbort(signal);
					},
				},
				{
					name: 'second',
					label: 'Second',
					description: 'Must not run after the abort.',
					parameters: { type: 'object', properties: {} },
					async execute() {
						calls.second += 1;
						return {
							details: {},
							content: [{ type: 'text' as const, text: 'second result' }],
						};
					},
				},
			],
		});
		return 'Run the supplied tool calls.';
	}

	const faux = fauxProvider({ models: [{ id: 'model' }] });
	faux.setResponses([
		fauxAssistantMessage(
			[
				fauxToolCall('first', {}, { id: 'call_first' }),
				fauxToolCall('second', {}, { id: 'call_second' }),
			],
			{ stopReason: 'toolUse' },
		),
		fauxAssistantMessage([fauxText('Recovered.')], { stopReason: 'stop' }),
	]);
	const database = recordingDatabase();
	const observedErrors: unknown[] = [];
	const disposeInstrumentation = instrument({
		dispose() {},
		observe() {},
		async interceptor(_operation, _context, next) {
			try {
				return await next();
			} catch (error) {
				observedErrors.push(error);
				throw error;
			}
		},
	});
	const runtime = await start({
		agents: [AbortBatch],
		db: database.adapter,
		providers: [faux.provider],
		env: {},
	});
	const agent = init(AbortBatch, { id: 'abort-partial-batch' });

	try {
		const receipt = await agent.dispatch('Run both tools.');
		await firstStarted.promise;
		await agent.abort();
		await expect(agent.read(receipt)).rejects.toMatchObject({
			name: 'AgentRunError',
			message: expect.stringMatching(/aborted/i),
		});
		expect(calls).toEqual({ first: 1, second: 0 });
		expect(
			observedErrors.filter(
				(error) => error instanceof Error && error.name === 'ConversationRecordInvariantError',
			),
		).toEqual([]);

		const outcomes = database.records.filter((record) => record.type === 'tool_outcome');
		expect(outcomes.map((record) => record.toolCallId)).toEqual(['call_first', 'call_second']);
		expect(outcomes[1]).toMatchObject({
			isError: true,
			content: [
				{
					type: 'text',
					text: expect.stringContaining('outcome is unknown'),
				},
			],
		});
		expect(
			database.records.filter((record) => record.type === 'tool_results_committed'),
		).toMatchObject([{ outcomeIds: outcomes.map((outcome) => outcome.id) }]);
		expect(database.records.filter((record) => record.type === 'state_write')).toMatchObject([
			{ value: 'started' },
		]);
		expect(
			database.records.find(
				(record) => record.type === 'signal' && record.signalType === 'submission_aborted',
			),
		).toMatchObject({
			attributes: {
				interruptedTools: JSON.stringify([{ name: 'second', id: 'call_second' }]),
			},
		});

		await expect(agent.read(await agent.dispatch('Continue.'))).resolves.toMatchObject({
			text: 'Recovered.',
		});
	} finally {
		await agent.abort();
		await runtime.stop();
		await disposeInstrumentation();
	}
});
