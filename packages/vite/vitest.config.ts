import { defineConfig } from 'vitest/config';

export default defineConfig({
	test: {
		// Some tests run a full Vite build. Under `turbo run test`, other packages
		// build and test in parallel, so the 5s default can expire.
		testTimeout: 30_000,
	},
});
