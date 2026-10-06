/**
 * The default provider set: every built-in. A separate module, so the whole
 * catalog enters only a build that uses the default — generated entries with
 * a configured `providers` list import the listed modules instead.
 */
export { registerDefaultProviders } from '../providers/all.ts';
