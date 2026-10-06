---
title: Provider API
description: Reference for the providers config, setProvider(), createProvider(), model resolution, and the Cloudflare AI binding provider in @flue/runtime.
lastReviewedAt: 2026-07-21
---

Flue's model layer runs on [TanStack AI](https://tanstack.com/ai). A provider is a Flue `Provider` object: the models it serves and the auth for a request. The runtime keeps one provider registry, and every model call resolves its model against it.

This page is the complete contract for:

- the [`providers` config](#the-providers-config), which selects the built-in providers that ship in a build
- [`setProvider()`](#setprovider), which registers a provider at runtime
- [`createProvider()`](#createprovider) and [`envApiKeyAuth()`](#envapikeyauth), which build a custom provider
- [`cloudflareBindingProvider()`](#cloudflarebindingprovider), which runs Workers AI models through `env.AI`

For a walkthrough, see [Models: Custom providers](/docs/guide/models/#custom-providers).

Exports:

```ts
import { createProvider, envApiKeyAuth, setProvider } from '@flue/runtime';
import { anthropicProvider } from '@flue/runtime/providers/anthropic';
import {
  cloudflareBindingProvider,
  type CloudflareAIBinding,
  type CloudflareBindingProviderOptions,
} from '@flue/runtime/cloudflare/workers-ai';
import { type CloudflareGatewayOptions } from '@flue/runtime/cloudflare';
```

## The `providers` config

```ts
// vite.config.ts
flue({ providers: ['anthropic', 'openai'] });
```

Selects which providers the generated server entry registers, by provider ID. Each entry becomes an import of `@flue/runtime/providers/<id>` in the generated entry. The exception is `'cloudflare'`, which selects Flue's own [Workers AI binding provider](#cloudflarebindingprovider). Only the listed providers and their model catalogs ship in the build. The IDs are in [Built-in providers](#built-in-providers).

- **Omitted: every built-in registers**, the Workers AI binding provider included on the Cloudflare target. The default keeps zero-config resolution: any `'provider/model-id'` specifier from the built-in catalog works, with credentials from the provider's environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, ...).
- **Set: the list is exhaustive.** A specifier naming an unlisted provider fails at [model resolution](#model-resolution), and on the Cloudflare target the binding provider is registered only when the list names `'cloudflare'`. Naming `'cloudflare'` on the Node target is a config error (the binding only exists on Workers). Custom providers are unaffected — register those with [`setProvider()`](#setprovider).
- **An unknown ID fails the build.** The generated import `@flue/runtime/providers/<id>` does not resolve, and the build error names it.
- **User registrations win.** The generated registrations skip any provider ID already registered, so a `setProvider()` in `app.ts` overrides a listed built-in regardless of module evaluation order.
- **[`flue run`](/docs/cli/run/) ignores the list.** It loads only the agent module — no `app.ts`, no generated entry — and always registers the full built-in set. The narrowing is a server-build concern.

The same field is accepted in `flue.config.ts`; inline plugin options win per field. See [Configuration](/docs/reference/configuration/#providers).

## `setProvider()`

```ts
function setProvider(provider: Provider): void;
```

Registers a `Provider` with the runtime, keyed by `provider.id`. After `setProvider(p)` with `p.id === 'acme'`, the specifier `'acme/some-model'` resolves through it. It accepts:

- a built-in factory, such as `anthropicProvider()` from `@flue/runtime/providers/anthropic`
- [`createProvider(...)`](#createprovider), for a custom endpoint
- [`cloudflareBindingProvider(...)`](#cloudflarebindingprovider)
- a faux provider's `.provider`, in tests (see [`start()`](/docs/reference/agent-api/#start))

```ts
import { createProvider, setProvider } from '@flue/runtime';

setProvider(
  createProvider({
    id: 'ollama',
    auth: {
      apiKey: { name: 'Ollama (keyless)', resolve: async () => ({ auth: {} }) },
    },
    models: [/* model records: each one has its own `api` and `baseUrl` */],
  }),
);
```

Behavior:

- **Each call replaces the ID's previous provider.** Calls do not accumulate or merge; the latest `setProvider()` for an ID wins, including over the generated entry's built-in registrations (which skip already-registered IDs).
- **The registry is module-scoped and in-memory.** Call `setProvider()` at module top level, before any agent runs. On the Node.js target one process hosts all agents, so a registration in `app.ts` covers everything. On the Cloudflare target each agent conversation runs in its own Durable Object isolate; `app.ts` is evaluated in every isolate, so top-level registrations apply everywhere. [`flue run`](/docs/cli/run/) loads only the agent module, never `app.ts` — put the registration in the agent module when it must also apply there.
- **Registration is declarative and deferred.** The call performs no network I/O and no credential validation; a wrong endpoint or key surfaces as a provider error on the first model request. There is no public unregister function.
- **Credentials resolve through the provider's own `auth`.** Each built-in provider reads its environment variables (see [Built-in providers](#built-in-providers)). A custom provider declares its own resolver: a fixed key, an environment read with [`envApiKeyAuth()`](#envapikeyauth), or a dynamic exchange. Flue adds no credential layer on top.

## Built-in providers

Each built-in provider has its own module, `@flue/runtime/providers/<id>`, which exports one factory (for example `anthropicProvider` from `@flue/runtime/providers/anthropic`). Its models come from the [`@tanstack/ai-models`](https://tanstack.com/ai) catalog. Its auth reads these environment variables:

| Provider ID                  | Credentials                                                                                                        |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `amazon-bedrock`             | `AWS_BEARER_TOKEN_BEDROCK`, or the AWS credential chain (`AWS_PROFILE`, access keys, and more)                     |
| `ant-ling`                   | `ANT_LING_API_KEY`                                                                                                 |
| `anthropic`                  | `ANTHROPIC_API_KEY`, `ANTHROPIC_OAUTH_TOKEN`, or `ANTHROPIC_AUTH_TOKEN`                                            |
| `azure-openai-responses`     | `AZURE_OPENAI_API_KEY`                                                                                             |
| `baseten`                    | `BASETEN_API_KEY`                                                                                                  |
| `cerebras`                   | `CEREBRAS_API_KEY`                                                                                                 |
| `cloudflare-ai-gateway`      | `CLOUDFLARE_API_KEY`, `CLOUDFLARE_ACCOUNT_ID`, and `CLOUDFLARE_GATEWAY_ID`                                         |
| `cloudflare-workers-ai`      | `CLOUDFLARE_API_KEY` and `CLOUDFLARE_ACCOUNT_ID`                                                                   |
| `deepseek`                   | `DEEPSEEK_API_KEY`                                                                                                 |
| `fireworks`                  | `FIREWORKS_API_KEY`                                                                                                |
| `google`                     | `GEMINI_API_KEY`                                                                                                   |
| `google-vertex`              | `GOOGLE_CLOUD_API_KEY`, or Application Default Credentials with `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` |
| `groq`                       | `GROQ_API_KEY`                                                                                                     |
| `huggingface`                | `HF_TOKEN`                                                                                                         |
| `kimi-coding`                | `KIMI_API_KEY`                                                                                                     |
| `meta`                       | `META_API_KEY`                                                                                                     |
| `minimax`                    | `MINIMAX_API_KEY`                                                                                                  |
| `minimax-cn`                 | `MINIMAX_CN_API_KEY`                                                                                               |
| `mistral`                    | `MISTRAL_API_KEY`                                                                                                  |
| `moonshotai`                 | `MOONSHOT_API_KEY`                                                                                                 |
| `moonshotai-cn`              | `MOONSHOT_API_KEY`                                                                                                 |
| `nvidia`                     | `NVIDIA_API_KEY`                                                                                                   |
| `openai`                     | `OPENAI_API_KEY`                                                                                                   |
| `opencode`                   | `OPENCODE_API_KEY`                                                                                                 |
| `opencode-go`                | `OPENCODE_API_KEY`                                                                                                 |
| `openrouter`                 | `OPENROUTER_API_KEY`                                                                                               |
| `qwen-token-plan`            | `QWEN_TOKEN_PLAN_API_KEY`                                                                                          |
| `qwen-token-plan-cn`         | `QWEN_TOKEN_PLAN_CN_API_KEY`                                                                                       |
| `qwen-token-plan-individual` | `QWEN_TOKEN_PLAN_API_KEY`                                                                                          |
| `together`                   | `TOGETHER_API_KEY`                                                                                                 |
| `vercel-ai-gateway`          | `AI_GATEWAY_API_KEY`                                                                                               |
| `xai`                        | `XAI_API_KEY`                                                                                                      |
| `xiaomi`                     | `XIAOMI_API_KEY`                                                                                                   |
| `xiaomi-token-plan-ams`      | `XIAOMI_TOKEN_PLAN_AMS_API_KEY`                                                                                    |
| `xiaomi-token-plan-cn`       | `XIAOMI_TOKEN_PLAN_CN_API_KEY`                                                                                     |
| `xiaomi-token-plan-sgp`      | `XIAOMI_TOKEN_PLAN_SGP_API_KEY`                                                                                    |
| `zai`                        | `ZAI_API_KEY`                                                                                                      |
| `zai-coding-cn`              | `ZAI_CODING_CN_API_KEY`                                                                                            |

`@flue/runtime/providers/all` exports `builtinProviders()`, which returns all of them.

## `createProvider()`

```ts
function createProvider(config: CreateProviderConfig): Provider;

interface CreateProviderConfig {
  id: string;
  name?: string; // default: the id
  baseUrl?: string;
  headers?: Record<string, string | null>;
  auth: { apiKey: ApiKeyAuth };
  models: FlueModel[];
  createAdapter?: (model: FlueModel, options: ModelAdapterOptions) => AnyTextAdapter;
}
```

Builds a provider from its models and its auth. Pass the result to [`setProvider()`](#setprovider). Exported from `@flue/runtime`.

- `models`: the model records. Each record names its own wire protocol (`api`) and endpoint (`baseUrl`), and carries its metadata. See [`FlueModel`](#fluemodel).
- `auth.apiKey`: the credential resolver. It runs before each model request. See [`envApiKeyAuth()`](#envapikeyauth).
- `headers`: headers for every request of this provider. A `null` value removes a header that the protocol adapter sends.
- `createAdapter`: optional. It builds the [TanStack AI](https://tanstack.com/ai) text adapter for a call itself, for a wire protocol that the `api` values below do not cover.

### `FlueModel`

```ts
interface FlueModel {
  id: string; // the model ID to send
  name: string;
  provider: string; // the provider ID
  api: string; // the wire protocol, see the table below
  baseUrl: string;
  input: ('text' | 'image' | 'document' | 'audio' | 'video')[];
  reasoning: boolean;
  reasoningMap?: Partial<Record<ThinkingLevel, string | null>>;
  cost: { input: number; output: number; cacheRead: number; cacheWrite: number }; // USD per 1M tokens
  contextWindow: number; // 0 means not known
  maxTokens: number; // 0 means not known
  headers?: Record<string, string>;
  compat?: ModelCompat; // request quirks of OpenAI-compatible endpoints
}
```

The `api` field picks the protocol adapter:

| `api`                     | Protocol                           |
| ------------------------- | ---------------------------------- |
| `anthropic-messages`      | Anthropic Messages                 |
| `openai-responses`        | OpenAI Responses                   |
| `openai-completions`      | OpenAI-compatible Chat Completions |
| `azure-openai-responses`  | Azure OpenAI Responses             |
| `google-generative-ai`    | Gemini API                         |
| `google-vertex`           | Gemini on Vertex AI                |
| `bedrock-converse-stream` | Amazon Bedrock Converse            |
| `mistral-conversations`   | Mistral                            |

The built-in providers use the same record shape. So `anthropicProvider().getModels()` gives records that you can copy and change.

## `envApiKeyAuth()`

```ts
function envApiKeyAuth(name: string, envVars: string[]): ApiKeyAuth;
```

API key auth from the first environment variable in `envVars` that has a value. `name` is a display name. Exported from `@flue/runtime`.

```ts
auth: { apiKey: envApiKeyAuth('Acme API key', ['ACME_API_KEY']) },
```

For any other credential, write the resolver yourself. `resolve` returns the auth for one request, or `undefined` when the provider has no credential:

```ts
interface ApiKeyAuth {
  name: string;
  resolve(input: { ctx: AuthContext; signal: AbortSignal }): Promise<AuthResult | undefined>;
}

interface AuthResult {
  auth: {
    apiKey?: string;
    headers?: Record<string, string | null>;
    baseUrl?: string;
  };
  source?: string; // where the credential came from, for status output
}

interface AuthContext {
  env(name: string): Promise<string | undefined>; // a blank value reads as undefined
  fileExists(path: string): Promise<boolean>; // a leading ~ is the home directory
}
```

## Model resolution

A model specifier is `'provider-id/model-id'`, split at the first `/`. Resolution happens per model call, against the live registry:

1. The provider ID must be registered — via the [`providers` config](#the-providers-config) defaults or [`setProvider()`](#setprovider). Unknown provider IDs throw, naming the registered IDs and both registration paths.
2. The model ID must be one the provider declares (`provider.getModels()`). Unknown model IDs throw, listing the declared IDs.
3. Exception: a provider with a dynamic-model template resolves model IDs that it does not declare. [`cloudflareBindingProvider()`](#cloudflarebindingprovider) is the only one Flue ships. Such a model gets zero metadata:
   - `reasoning: false`: a forwarded `thinkingLevel` is dropped.
   - `input: ['text']`: image blocks are replaced with an `"(image omitted)"` placeholder.
   - `contextWindow: 0` and `maxTokens: 0`: not known, so threshold [compaction](/docs/guide/models/#compaction) cannot start.
   - all-zero cost.

   A gateway-vendor ID (`anthropic/...`, `openai/...`) that uses this fallback logs a one-time warning that names the model and the lost metadata. The usual fix is a newer Flue release whose catalog has the model. `@cf/...` IDs resolve this way with no warning, because their wire format needs no catalog record.

Resolution failures throw plain `Error`s (not `FlueError` categories), raised when the model call resolves the specifier. A specifier with no `/`, or with an empty model ID (`'acme/'`), is invalid.

Model metadata (context window, cost, reasoning support, input modalities, per-model headers) lives on the model records that the provider declares. To change the metadata of a built-in provider, register a replacement provider whose records have the values you need (see [Models: Custom providers](/docs/guide/models/#custom-providers)). There is no separate override surface.

## `cloudflareBindingProvider()`

```ts
function cloudflareBindingProvider(options: CloudflareBindingProviderOptions): Provider;

interface CloudflareBindingProviderOptions {
  binding: CloudflareAIBinding; // env.AI
  gateway?: CloudflareGatewayOptions | false;
  streamIdleTimeoutMs?: number;
}
```

Builds the `cloudflare` provider. Its models run through a [Workers AI binding](https://developers.cloudflare.com/workers-ai/)'s `run(modelId, payload, options)` in the same process: no `baseUrl`, no `apiKey`, no HTTP endpoint. The provider keeps the binding and the resolved gateway options. The factory has its own subpath, `@flue/runtime/cloudflare/workers-ai`, so the binding code enters a build only when you import it. It is not part of the `@flue/runtime/cloudflare` barrel.

- `binding` — the captured `env.AI` reference.
- `gateway` — [AI Gateway](https://developers.cloudflare.com/ai-gateway/) routing for every `run` call through this provider. Tri-state: omitted routes through Cloudflare's default AI Gateway (the options object `{ id: 'default' }`, which the binding provisions on demand for the account); a [`CloudflareGatewayOptions`](#cloudflaregatewayoptions) object replaces the default; `false` opts out — no gateway option is passed to `run`.
- `streamIdleTimeoutMs` — cap on how long a model stream may go without delivering a byte before the request fails as a retryable interruption, which the turn retries under the transient-error budget. Defaults to five minutes — generous, because a long-thinking model can be legitimately silent when neither keepalives nor reasoning deltas stream. `0` disables the guard.

Behavior:

- **Registration on the Cloudflare target.** When the [`providers` config](#the-providers-config) is omitted or names `'cloudflare'`, the generated Worker entry runs `setProvider(cloudflareBindingProvider({ binding: env.AI }))` — unless a provider with the `cloudflare` ID is already registered. `app.ts` imports are hoisted above the generated entry's body, so a user registration always wins; this is how a project targets a named gateway, tunes caching, or opts out. A `providers` list without `'cloudflare'` emits neither the registration nor the import. See [Models — Cloudflare Workers AI](/docs/guide/models/#cloudflare-workers-ai-cloudflare-only) and the [Cloudflare target guide](/docs/guide/cloudflare-target/#workers-ai-and-ai-gateway).
- **Metadata.** The provider declares the `cloudflare-workers-ai` catalog under the `cloudflare` ID, and adds the `cloudflare-ai-gateway` catalog. A gateway model ID gets the vendor of its gateway URL as a prefix (`gpt-5.6-terra` becomes `openai/gpt-5.6-terra`), which is how the binding addresses it. So gateway models have their real context window, cost, reasoning support (`reasoningMap`), input modalities, and wire protocol (`api`). Gateway `/compat` records are skipped: they repeat `@cf/...` IDs that the Workers AI catalog already declares. Any other ID still resolves, because the binding accepts any model ID, with the zero-metadata defaults from [Model resolution](#model-resolution).
- **Wire format.** Each model gets one serialization:
  1. The catalog `api` comes first.
  2. For an ID that no catalog has, the vendor prefix decides: `anthropic/...` uses Anthropic Messages, and `openai/...` uses OpenAI Responses.
  3. `@cf/...` IDs and unknown vendors use the OpenAI-compatible Chat Completions shape.

  Anthropic and OpenAI models run through the TanStack AI Anthropic and OpenAI adapters, and `@cf/...` models through `@tanstack/ai-cloudflare`. Reasoning effort maps through the model's catalog `reasoningMap`. A model whose catalog `api` has no binding path fails with a stream error, not with a wrong payload. Every request goes through `binding.run` with `returnRawResponse: true` and the resolved `gateway` option. The request body has no `model` field: the first `run()` argument names the model.

- **Failures.** Non-OK binding responses throw `CloudflareAIBindingError` (`type: 'cloudflare_ai_binding_error'`), exported from `@flue/runtime/cloudflare`; a 413 additionally carries `meta.reason: 'request_too_large'` and triggers compaction recovery. See [Errors — `CloudflareAIBindingError`](/docs/reference/errors/#cloudflareaibindingerror).
- **Node import safety.** The factory and its types are importable on Node.js (the binding shape is structural); only calling a model through it requires a real binding.

## `CloudflareGatewayOptions`

```ts
interface CloudflareGatewayOptions {
  id: string;
  skipCache?: boolean;
  cacheTtl?: number;
  cacheKey?: string;
  metadata?: Record<string, number | string | boolean | null | bigint>;
  collectLog?: boolean;
  eventId?: string;
  requestTimeoutMs?: number;
}
```

AI Gateway options applied to every `binding.run(...)` call routed through a [`cloudflareBindingProvider()`](#cloudflarebindingprovider). The shape mirrors [Cloudflare's Worker binding methods documentation](https://developers.cloudflare.com/ai-gateway/integrations/worker-binding-methods/), which defines each field's provider-side semantics; every field except `requestTimeoutMs` is forwarded verbatim in the `gateway` option. Exported from `@flue/runtime/cloudflare`.

- `id` — the AI Gateway ID (slug) to route through. Required whenever gateway options are specified.
- `skipCache` — bypass the gateway cache for the request.
- `cacheTtl` — cache TTL override, in seconds.
- `cacheKey` — cache key override.
- `metadata` — arbitrary metadata surfaced on the gateway log entry.
- `collectLog` — force collecting (or not collecting) request logs.
- `eventId` — custom event ID for log correlation.
- `requestTimeoutMs` — gateway-enforced bound, in milliseconds, on the time to the first part of the response (not total duration — pair with [`streamIdleTimeoutMs`](#cloudflarebindingprovider) for mid-stream stalls). The binding has no equivalent field, so this is emitted as the `cf-aig-request-timeout` header on the request rather than forwarded in the `gateway` option.

## `CloudflareAIBinding`

```ts
interface CloudflareAIBinding {
  run(
    modelId: string,
    inputs: Record<string, unknown>,
    options?: Record<string, unknown>,
  ): Promise<Response | Record<string, unknown>>;
}
```

The minimal structural shape of a Workers AI binding, exported from `@flue/runtime/cloudflare`. It is deliberately structural — not Cloudflare's `Ai` type — so the factory stays importable on Node.js. Pass the real `env.AI` binding as [`CloudflareBindingProviderOptions`](#cloudflarebindingprovider)' `binding`.

## Provider telemetry

[Model-turn observability events](/docs/reference/events/#turn_start-turn_request-turn-turn_messages) (`turn_request.request` and `turn.request`) identify the provider with a fixed normalization of the provider ID to observability conventions (`ModelRequestInfo.providerName`); IDs outside the table pass through unchanged. The reported server host and port are parsed from the resolved model's `baseUrl`.

| Provider ID                   | `providerName`    |
| ----------------------------- | ----------------- |
| `amazon-bedrock`              | `aws.bedrock`     |
| `azure-openai-responses`      | `azure.ai.openai` |
| `google`                      | `gcp.gemini`      |
| `google-vertex`               | `gcp.vertex_ai`   |
| `mistral`                     | `mistral_ai`      |
| `moonshotai`, `moonshotai-cn` | `moonshot_ai`     |
| `xai`                         | `x_ai`            |

The same events always report the provider ID unmodified as `ModelRequestInfo.providerId`.

## What registration does not change

- **The built-in catalog.** A registration replaces one provider ID at resolution time. It never adds, removes, or changes catalog records.
- **Anything durable.** Registrations live in process memory for the current module scope. They are not persisted, not shared across processes or isolates, and rebuilt from module top-level code on every boot.
- **Earlier registrations of other provider IDs.** Each call affects exactly one provider ID.
