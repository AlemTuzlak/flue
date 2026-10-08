import { afterEach, describe, expect, it, vi } from 'vitest';
import { cloudflareAIGatewayProvider } from '../providers/cloudflare-ai-gateway.ts';
import {
	createProvider,
	type DynamicModelTemplate,
	envApiKeyAuth,
	type FlueModel,
} from '../providers/provider.ts';
import {
	anthropicGatewayModelId,
	DYNAMIC_MODEL_MARKER,
	getProvider,
	isAnthropicGatewayModel,
	isDynamicModel,
	resetDynamicModelWarnForTests,
	resetModelsForTests,
	resetVersionSeparatorAliasWarnForTests,
	resolveModel,
	setProvider,
} from './providers.ts';

const ANY_ENDPOINT: DynamicModelTemplate = {
	api: 'anthropic-messages',
	baseUrl: 'https://example.test',
};

function providerWith(
	providerId: string,
	models: FlueModel[],
	dynamicModels?: DynamicModelTemplate,
) {
	return createProvider({
		id: providerId,
		baseUrl: 'https://example.test',
		auth: { apiKey: envApiKeyAuth('Test API key', ['TEST_API_KEY']) },
		models,
		...(dynamicModels ? { dynamicModels } : {}),
	});
}

function registeredIds(providerId: string) {
	return getProvider(providerId)
		?.getModels()
		.map((model) => model.id);
}

afterEach(() => {
	resetModelsForTests();
	resetDynamicModelWarnForTests();
	resetVersionSeparatorAliasWarnForTests();
	vi.restoreAllMocks();
});

describe('dynamic model templates', () => {
	it('synthesizes a model marked as dynamic for ids no catalog knows', () => {
		vi.spyOn(console, 'warn').mockImplementation(() => {});
		setProvider(providerWith('test', [], ANY_ENDPOINT));
		const model = resolveModel('test/fresh-model');
		expect(model.id).toBe('fresh-model');
		expect(isDynamicModel(model)).toBe(true);
		expect(Reflect.get(model, DYNAMIC_MODEL_MARKER)).toBe(true);
	});

	it('keeps a zero cost table so usage can be computed', () => {
		vi.spyOn(console, 'warn').mockImplementation(() => {});
		setProvider(providerWith('test', [], ANY_ENDPOINT));
		const model = resolveModel('test/fresh-model');
		expect(model.cost).toEqual({
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
		});
		// `shouldCompact` treats a non-positive window as unknown.
		expect(model.contextWindow).toBe(0);
		expect(model.maxTokens).toBe(0);
	});

	it('does not mark catalog models', () => {
		setProvider(providerWith('test', [catalogModel('test', 'known-model')]));
		const model = resolveModel('test/known-model');
		expect(isDynamicModel(model)).toBe(false);
		expect(Reflect.get(model, DYNAMIC_MODEL_MARKER)).toBeUndefined();
	});

	it('warns once per process when the template is used', () => {
		const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
		setProvider(providerWith('test', [], ANY_ENDPOINT));
		resolveModel('test/first-unknown');
		resolveModel('test/second-unknown');
		expect(warn).toHaveBeenCalledTimes(1);
		expect(warn.mock.calls[0]?.[0]).toContain('test/first-unknown');
		expect(warn.mock.calls[0]?.[0]).toContain('isDynamicModel');
	});
});

function catalogModel(providerId: string, id: string): FlueModel {
	return {
		id,
		name: id,
		api: 'anthropic-messages',
		provider: providerId,
		baseUrl: 'https://example.test',
		reasoning: false,
		input: ['text'],
		cost: { input: 1, output: 2, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 1000,
		maxTokens: 500,
	};
}

describe('version-separator aliases', () => {
	it('resolves a dashed version to the dotted catalog id', () => {
		vi.spyOn(console, 'warn').mockImplementation(() => {});
		setProvider(providerWith('test', [catalogModel('test', 'claude-sonnet-4.6')]));
		const model = resolveModel('test/claude-sonnet-4-6');
		expect(model.id).toBe('claude-sonnet-4.6');
		expect(isDynamicModel(model)).toBe(false);
	});

	it('resolves a dotted version to the dashed catalog id', () => {
		vi.spyOn(console, 'warn').mockImplementation(() => {});
		setProvider(providerWith('test', [catalogModel('test', 'claude-sonnet-4-6')]));
		expect(resolveModel('test/claude-sonnet-4.6').id).toBe('claude-sonnet-4-6');
	});

	it('prefers an exact match over an alias', () => {
		const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
		setProvider(
			providerWith('test', [
				catalogModel('test', 'claude-3.5-haiku'),
				catalogModel('test', 'claude-3-5-haiku'),
			]),
		);
		expect(resolveModel('test/claude-3-5-haiku').id).toBe('claude-3-5-haiku');
		expect(warn).not.toHaveBeenCalled();
	});

	it('does not alias when several catalog ids match', () => {
		setProvider(
			providerWith('test', [
				catalogModel('test', 'model-1.2-3'),
				catalogModel('test', 'model-1-2.3'),
			]),
		);
		expect(() => resolveModel('test/model-1-2-3')).toThrow('Unknown model ID');
	});

	it('does not alias ids that differ beyond version separators', () => {
		setProvider(providerWith('test', [catalogModel('test', 'claude-opus-5')]));
		expect(() => resolveModel('test/claude-opus-5-5')).toThrow('Unknown model ID');
	});

	it('prefers an alias over dynamic synthesis', () => {
		vi.spyOn(console, 'warn').mockImplementation(() => {});
		setProvider(providerWith('test', [catalogModel('test', 'claude-sonnet-4.6')], ANY_ENDPOINT));
		const model = resolveModel('test/claude-sonnet-4-6');
		expect(model.id).toBe('claude-sonnet-4.6');
		expect(isDynamicModel(model)).toBe(false);
	});

	it('warns once per requested specifier', () => {
		const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
		setProvider(providerWith('test', [catalogModel('test', 'claude-sonnet-4.6')]));
		resolveModel('test/claude-sonnet-4-6');
		resolveModel('test/claude-sonnet-4-6');
		expect(warn).toHaveBeenCalledTimes(1);
		expect(warn.mock.calls[0]?.[0]).toContain('test/claude-sonnet-4.6');
	});
});

const GATEWAY_BASE =
	'https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}';

function gatewayModel(id: string, api: string, vendor: string): FlueModel {
	return {
		...catalogModel('cloudflare-ai-gateway', id),
		api,
		baseUrl: `${GATEWAY_BASE}/${vendor}`,
	};
}

describe('Cloudflare AI Gateway Anthropic ids', () => {
	it('dashes dotted versions', () => {
		expect(anthropicGatewayModelId('claude-sonnet-4.6')).toBe('claude-sonnet-4-6');
		expect(anthropicGatewayModelId('claude-fable-5.1')).toBe('claude-fable-5-1');
		expect(anthropicGatewayModelId('claude-opus-5')).toBe('claude-opus-5');
		expect(anthropicGatewayModelId('claude-sonnet-4-6')).toBe('claude-sonnet-4-6');
	});

	it('registers native Anthropic endpoint models under Anthropic ids', () => {
		setProvider(
			providerWith('cloudflare-ai-gateway', [
				gatewayModel('claude-sonnet-4.6', 'anthropic-messages', 'anthropic'),
				gatewayModel('claude-opus-5', 'anthropic-messages', 'anthropic'),
				gatewayModel('gpt-5.6-terra', 'openai-responses', 'openai'),
				gatewayModel('workers-ai/@cf/moonshotai/kimi-k2.6', 'openai-completions', 'compat'),
			]),
		);
		expect(registeredIds('cloudflare-ai-gateway')).toEqual([
			'claude-sonnet-4-6',
			'claude-opus-5',
			'gpt-5.6-terra',
			'workers-ai/@cf/moonshotai/kimi-k2.6',
		]);
		expect(resolveModel('cloudflare-ai-gateway/claude-sonnet-4-6').id).toBe('claude-sonnet-4-6');
	});

	it('resolves dotted specifiers to the Anthropic id', () => {
		vi.spyOn(console, 'warn').mockImplementation(() => {});
		setProvider(
			providerWith('cloudflare-ai-gateway', [
				gatewayModel('claude-sonnet-4.6', 'anthropic-messages', 'anthropic'),
			]),
		);
		expect(resolveModel('cloudflare-ai-gateway/claude-sonnet-4.6').id).toBe('claude-sonnet-4-6');
	});

	it('keeps one entry when the catalog lists both forms', () => {
		setProvider(
			providerWith('cloudflare-ai-gateway', [
				gatewayModel('claude-sonnet-4.6', 'anthropic-messages', 'anthropic'),
				gatewayModel('claude-sonnet-4-6', 'anthropic-messages', 'anthropic'),
			]),
		);
		expect(registeredIds('cloudflare-ai-gateway')).toEqual(['claude-sonnet-4-6']);
	});

	it('leaves other providers unchanged', () => {
		setProvider(
			providerWith('cloudflare', [
				{
					...gatewayModel('anthropic/claude-sonnet-4.6', 'anthropic-messages', 'anthropic'),
					provider: 'cloudflare',
				},
			]),
		);
		expect(resolveModel('cloudflare/anthropic/claude-sonnet-4.6').id).toBe(
			'anthropic/claude-sonnet-4.6',
		);
	});

	it('leaves no dotted Anthropic ids in the shipped gateway catalog', () => {
		setProvider(cloudflareAIGatewayProvider());
		const anthropic = (getProvider('cloudflare-ai-gateway')?.getModels() ?? []).filter(
			isAnthropicGatewayModel,
		);
		expect(anthropic.length).toBeGreaterThan(0);
		expect(anthropic.filter((model) => /\d\.\d/.test(model.id)).map((model) => model.id)).toEqual(
			[],
		);
		expect(resolveModel('cloudflare-ai-gateway/claude-sonnet-4-6').id).toBe('claude-sonnet-4-6');
	});
});
