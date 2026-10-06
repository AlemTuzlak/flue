import { describe, expect, it } from 'vitest';
import type { AssistantMessage } from './llm-types.ts';
import { isRetryableModelError } from './submission-state.ts';

function failedMessage(errorMessage: string): AssistantMessage {
	return {
		role: 'assistant',
		content: [],
		api: 'anthropic-messages',
		provider: 'anthropic',
		model: 'claude-sonnet-4-5',
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: 'error',
		errorMessage,
		timestamp: 1,
	};
}

describe('isRetryableModelError', () => {
	it('retries a failed assistant message with a transient error', () => {
		expect(isRetryableModelError(failedMessage('upstream server error: overloaded'))).toBe(true);
	});

	it('does not retry a failed assistant message with a request error', () => {
		expect(isRetryableModelError(failedMessage('Invalid tool schema'))).toBe(false);
	});

	it('reads a TanStack model error by its message', () => {
		expect(isRetryableModelError({ message: 'Too Many Requests' })).toBe(true);
	});

	it('reads a TanStack model error by its code', () => {
		expect(isRetryableModelError({ message: 'Request failed', code: '503' })).toBe(true);
	});

	it('does not retry a TanStack model error with a request error', () => {
		expect(isRetryableModelError({ message: 'Request aborted', code: 'aborted' })).toBe(false);
	});
});
