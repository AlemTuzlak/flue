import { expect, it } from 'vitest';
import { init, instrument, useModel } from './index.ts';
import { sqlite, start } from './node/index.ts';
import { fauxAssistantMessage, fauxProvider, fauxText } from './test-utils/faux.ts';
import type { FlueObservation } from './types.ts';

it('includes final text in persisted submission operation telemetry', async () => {
	function ObservedAgent() {
		useModel('faux/model');
		return 'Reply to the user.';
	}

	const faux = fauxProvider({ models: [{ id: 'model' }] });
	faux.setResponses([
		fauxAssistantMessage([fauxText('Persisted response.')], {
			stopReason: 'stop',
		}),
	]);
	const observations: FlueObservation[] = [];
	const disposeInstrumentation = instrument({
		dispose() {},
		observe(event) {
			observations.push(event);
		},
		interceptor(_operation, _context, next) {
			return next();
		},
	});
	const runtime = await start({
		agents: [ObservedAgent],
		db: sqlite(),
		providers: [faux.provider],
		env: {},
	});
	const agent = init(ObservedAgent, { id: 'persisted-output' });

	try {
		await expect(agent.read(await agent.dispatch('Hello.'))).resolves.toMatchObject({
			text: 'Persisted response.',
		});
		expect(
			observations.findLast(
				(event) => event.type === 'operation' && event.operationKind === 'prompt' && !event.isError,
			),
		).toMatchObject({
			agentOutput: {
				type: 'text',
				text: 'Persisted response.',
				finishReason: 'stop',
			},
		});
	} finally {
		await agent.abort();
		await runtime.stop();
		await disposeInstrumentation();
	}
});
