/**
 * The runtime's providers: Flue providers on `@tanstack/ai-models`. `app.ts`,
 * the generated server entries, and `flue run` register them, and
 * `resolveModel` reads them for each model call.
 */
import { resetProvidersForTests } from '../providers/registry.ts';

export {
	DYNAMIC_MODEL_MARKER,
	isDynamicModel,
	resetDynamicModelWarnForTests,
} from '../providers/catalog.ts';
export {
	anthropicGatewayModelId,
	getProvider,
	hasProvider,
	isAnthropicGatewayModel,
	registerBuiltinProviderModule,
	resetVersionSeparatorAliasWarnForTests,
	resolveModel,
	setProvider,
} from '../providers/registry.ts';

/** Forget every registered provider. Test-only. */
export const resetModelsForTests = resetProvidersForTests;

// ─── Telemetry naming ───────────────────────────────────────────────────────

/**
 * OpenTelemetry GenAI system name for a provider ID, per the semconv
 * `gen_ai.system` well-known values. Unlisted IDs pass through unchanged.
 */
export function providerTelemetryName(providerId: string): string {
	return (
		{
			'amazon-bedrock': 'aws.bedrock',
			anthropic: 'anthropic',
			'azure-openai-responses': 'azure.ai.openai',
			deepseek: 'deepseek',
			google: 'gcp.gemini',
			'google-vertex': 'gcp.vertex_ai',
			groq: 'groq',
			mistral: 'mistral_ai',
			moonshotai: 'moonshot_ai',
			'moonshotai-cn': 'moonshot_ai',
			openai: 'openai',
			xai: 'x_ai',
		}[providerId] ?? providerId
	);
}
