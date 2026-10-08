import { expect, it } from 'vitest';
import { init, instrument, useModel } from './index.ts';
import { sqlite, start } from './node/index.ts';
import { fauxAssistantMessage, fauxProvider, fauxText } from './test-utils/faux.ts';
import type { FlueObservation } from './types.ts';

const longMessage = 'x'.repeat(70_000);

it('settles a completed response after silent-overflow compaction', async () => {
	function OverflowAgent() {
		useModel('faux/model', { compaction: false });
		return 'Reply to the user.';
	}

	const faux = fauxProvider({
		models: [{ id: 'model', contextWindow: 32_768, maxTokens: 4_096 }],
	});
	faux.setResponses([
		fauxAssistantMessage([fauxText('First response.')], { stopReason: 'stop' }),
		fauxAssistantMessage([fauxText('Completed response.')], {
			stopReason: 'stop',
		}),
		fauxAssistantMessage([fauxText('Conversation summary.')], {
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
		agents: [OverflowAgent],
		db: sqlite(),
		providers: [faux.provider],
		env: {},
	});
	const agent = init(OverflowAgent, { id: 'completed-overflow' });

	try {
		await agent.read(await agent.dispatch(longMessage));
		await expect(agent.read(await agent.dispatch(longMessage))).resolves.toMatchObject({
			text: 'Completed response.',
		});
		expect(faux.state.callCount).toBe(3);
		expect(observations.some((event) => event.type === 'compaction' && !event.isError)).toBe(true);
	} finally {
		await agent.abort();
		await runtime.stop();
		await disposeInstrumentation();
	}
});

it('still retries error-based overflow after compaction', async () => {
	function OverflowAgent() {
		useModel('faux/model', { compaction: false });
		return 'Reply to the user.';
	}

	const faux = fauxProvider({
		models: [{ id: 'model', contextWindow: 32_768, maxTokens: 4_096 }],
	});
	faux.setResponses([
		fauxAssistantMessage([fauxText('First response.')], { stopReason: 'stop' }),
		fauxAssistantMessage([], {
			stopReason: 'error',
			errorMessage: 'Request exceeds the context window.',
		}),
		fauxAssistantMessage([fauxText('Conversation summary.')], {
			stopReason: 'stop',
		}),
		fauxAssistantMessage([fauxText('Recovered response.')], {
			stopReason: 'stop',
		}),
	]);
	const runtime = await start({
		agents: [OverflowAgent],
		db: sqlite(),
		providers: [faux.provider],
		env: {},
	});
	const agent = init(OverflowAgent, { id: 'error-overflow' });

	try {
		await agent.read(await agent.dispatch(longMessage));
		await expect(agent.read(await agent.dispatch(longMessage))).resolves.toMatchObject({
			text: 'Recovered response.',
		});
		expect(faux.state.callCount).toBe(4);
	} finally {
		await agent.abort();
		await runtime.stop();
	}
});
