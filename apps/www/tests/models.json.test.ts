/**
 * Smoke test for the `models.json` model-discovery endpoint: the catalog Flue
 * ships on https://flueframework.com/models.json must actually contain the
 * model records this project advertises. If the endpoint's catalog pin lags
 * the runtime's, the discovery list silently omits models that work.
 */
import { getModels, getProviders } from '@tanstack/ai-models';
import { describe, expect, it } from 'vitest';

function endpointSpecifiers(): string[] {
	const modelSpecifiers = getProviders().flatMap((provider) =>
		getModels(provider.id).map((model) => `${provider.id}/${model.id}`),
	);
	if (modelSpecifiers.length === 0) {
		throw new Error('No model specifiers found in the @tanstack/ai-models catalog.');
	}
	return modelSpecifiers;
}

describe('models.json endpoint catalog', () => {
	it('contains the GPT-6 Sol/Luna and Claude Opus 5.5 records', () => {
		const specifiers = endpointSpecifiers();
		for (const expected of ['openai/gpt-6-sol', 'openai/gpt-6-luna', 'anthropic/claude-opus-5-5']) {
			expect(specifiers, `catalog should include ${expected}`).toContain(expected);
		}
	});

	it('does not ship an empty catalog', () => {
		expect(endpointSpecifiers().length).toBeGreaterThan(0);
	});
});
