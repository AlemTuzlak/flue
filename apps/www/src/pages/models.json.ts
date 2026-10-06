import { getModels, getProviders } from '@tanstack/ai-models';
import type { APIRoute } from 'astro';

export const GET: APIRoute = () => {
	const modelSpecifiers = getProviders().flatMap((provider) =>
		getModels(provider.id).map((model) => `${provider.id}/${model.id}`),
	);

	if (modelSpecifiers.length === 0) {
		throw new Error('No model specifiers found in the @tanstack/ai-models catalog.');
	}

	return new Response(JSON.stringify(modelSpecifiers, null, 2), {
		headers: {
			'Content-Type': 'application/json; charset=utf-8',
		},
	});
};
