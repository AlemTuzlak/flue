/**
 * Models that no catalog lists. A provider with `dynamicModels` serves any
 * model id, and Flue gives such a model zero metadata and a marker.
 */
import type { DynamicModelTemplate } from './provider.ts';

/**
 * Marks a model made from a dynamic model template. Its cost reads as zero and
 * its context window as `0`, which means "not known", not "free" or "empty".
 */
export const DYNAMIC_MODEL_MARKER = Symbol.for('flue.dynamicModelMarker');

/** True when `model` was made from a dynamic model template. */
export function isDynamicModel(model: object) {
	return DYNAMIC_MODEL_MARKER in model && model[DYNAMIC_MODEL_MARKER] === true;
}

/** One warning per process: the template was used. */
let warnedAboutDynamicModel = false;

/** Warn once per process that a model was made from a dynamic model template. */
export function warnDynamicModel(providerId: string, modelId: string) {
	if (warnedAboutDynamicModel) return;
	warnedAboutDynamicModel = true;
	console.warn(
		`[flue] Model "${providerId}/${modelId}" is not in the provider's catalog and was ` +
			`synthesized from a dynamic model template. Its cost reads as $0 and it has no known ` +
			`context window — detect such models with isDynamicModel() from "@flue/runtime".`,
	);
}

/** Reset the one-time dynamic model warning. Test-only. */
export function resetDynamicModelWarnForTests() {
	warnedAboutDynamicModel = false;
}

/** A zero-metadata model for an id that the provider does not list. */
export function dynamicModel(providerId: string, modelId: string, template: DynamicModelTemplate) {
	warnDynamicModel(providerId, modelId);
	return {
		id: modelId,
		name: modelId,
		api: template.api,
		provider: providerId,
		baseUrl: template.baseUrl,
		reasoning: false,
		input: ['text'] as const,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		// `0` reads as "not known", so threshold compaction does not engage.
		contextWindow: 0,
		maxTokens: 0,
		[DYNAMIC_MODEL_MARKER]: true,
	};
}
