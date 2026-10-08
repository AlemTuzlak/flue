import {
	type AuthProvider,
	SSEClientTransport,
	StreamableHTTPClientTransport,
	type Transport,
} from '@modelcontextprotocol/client';
import type { ContentPart } from '@tanstack/ai';
import { createMCPClient, type McpServerTool } from '@tanstack/ai-mcp';
import { version as runtimeVersion } from '../package.json' with { type: 'json' };
import type { ImageContent, TextContent } from './llm-types.ts';
import type { McpAuth, McpConnectionDefinition, McpTransport } from './mcp-types.ts';
import { registerPreparedToolAdapter } from './tool-adapter.ts';
import type { ToolDefinition } from './types.ts';

export type {
	McpAuth,
	McpConnectionDefinition,
	McpToolAnnotations,
	McpTransport,
} from './mcp-types.ts';

/** Connection returned by {@link createMcpConnection}. */
export interface McpConnection {
	/** Server name supplied to {@link createMcpConnection}. */
	name: string;
	/** MCP tools adapted into ordinary Flue tool definitions. */
	tools: ToolDefinition[];
	/** Close the underlying MCP client connection. */
	close(): Promise<void>;
}

/**
 * Resolves `useMcpConnection()` declarations to live connections.
 * Coordinators inject a per-instance caching resolver; a context without one
 * connects fresh at every harness initialization.
 */
export interface McpConnectionResolver {
	resolve(definition: McpConnectionDefinition): Promise<McpConnection>;
}

/** A caching {@link McpConnectionResolver} with a teardown for coordinator shutdown. */
export interface McpConnectionCache extends McpConnectionResolver {
	/** Close every cached connection and forget them all. */
	close(): Promise<void>;
}

/**
 * A per-instance MCP connection cache: the first declaration of a server
 * name connects; later submissions reuse the live connection for the
 * instance's in-memory lifetime, so definitions are read at first connect
 * (an `auth` resolver stays per-request). Concurrent resolves of one name
 * share a single in-flight connect. A failed connect is evicted immediately —
 * a transient outage must not brick the instance, so the next submission
 * retries with a freshly read definition.
 */
export function createMcpConnectionCache(): McpConnectionCache {
	const connections = new Map<string, Promise<McpConnection>>();
	return {
		resolve(definition: McpConnectionDefinition): Promise<McpConnection> {
			const cached = connections.get(definition.name);
			if (cached) return cached;
			const pending = createMcpConnection(definition);
			connections.set(definition.name, pending);
			pending.catch(() => {
				if (connections.get(definition.name) === pending) {
					connections.delete(definition.name);
				}
			});
			return pending;
		},
		async close(): Promise<void> {
			const pending = [...connections.values()];
			connections.clear();
			await Promise.allSettled(pending.map(async (connection) => (await connection).close()));
		},
	};
}

/**
 * Connects to a remote MCP server described by a
 * {@link McpConnectionDefinition} and adapts its listed tools into ordinary
 * Flue tool definitions.
 *
 * Adapted tool names use `mcp__<server>__<tool>`. Unsupported characters are
 * replaced with underscores, and duplicate adapted names are rejected. Close
 * the returned connection when its tools are no longer needed.
 */
export async function createMcpConnection(
	definition: McpConnectionDefinition,
): Promise<McpConnection> {
	const url = definition.url instanceof URL ? definition.url : new URL(definition.url);
	const transport = createTransport(
		url,
		definition.transport ?? 'streamable-http',
		mergeRequestInit(definition.requestInit, definition.headers),
		definition.fetch,
		definition.auth === undefined ? undefined : createAuthProvider(definition.auth),
	);
	return createMcpConnectionFromTransport(definition.name, transport, {
		timeoutMs: definition.timeoutMs,
		resetTimeoutOnProgress: definition.resetTimeoutOnProgress,
		tools: definition.tools,
	});
}

/**
 * Connects an MCP client over `transport` and adapts the server's tools.
 * `tools` is a strict allowlist: a name the server does not expose fails the
 * connection.
 */
export async function createMcpConnectionFromTransport(
	name: string,
	transport: Transport,
	options: Pick<McpConnectionDefinition, 'timeoutMs' | 'resetTimeoutOnProgress' | 'tools'> = {},
): Promise<McpConnection> {
	const client = await createMCPClient({
		transport,
		name: 'flue',
		version: runtimeVersion,
		toolName: (tool) => createToolName(name, tool.name),
		requestOptions: {
			timeout: options.timeoutMs,
			resetTimeoutOnProgress: options.resetTimeoutOnProgress,
		},
		...(options.tools === undefined ? {} : { toolFilter: options.tools }),
	});
	try {
		const tools = await client.tools();
		return {
			name,
			tools: tools.map((tool) => toFlueTool(tool)),
			close: () => client.close(),
		};
	} catch (error) {
		await client.close().catch(() => undefined);
		throw error;
	}
}

/**
 * Adapt the `auth` credential to the MCP SDK's {@link AuthProvider}: the
 * transport calls `token()` before every request, and on a 401 awaits
 * `onUnauthorized` and retries once — re-resolving the token, so the
 * application's credential store is the refresh policy.
 */
function createAuthProvider(auth: McpAuth): AuthProvider {
	const resolveToken = typeof auth === 'function' ? auth : () => auth;
	return {
		token: async () => resolveToken(),
		onUnauthorized: async () => {},
	};
}

function createTransport(
	url: URL,
	transport: McpTransport,
	requestInit: RequestInit,
	fetchImpl: typeof fetch | undefined,
	authProvider: AuthProvider | undefined,
) {
	if (transport === 'sse') {
		return new SSEClientTransport(url, {
			requestInit,
			fetch: fetchImpl,
			authProvider,
		});
	}
	return new StreamableHTTPClientTransport(url, {
		requestInit,
		fetch: fetchImpl,
		authProvider,
	});
}

function mergeRequestInit(
	requestInit: RequestInit | undefined,
	headers: HeadersInit | undefined,
): RequestInit {
	if (!headers) return requestInit ?? {};
	const mergedHeaders = new Headers(requestInit?.headers);
	for (const [key, value] of new Headers(headers)) {
		mergedHeaders.set(key, value);
	}
	return {
		...requestInit,
		headers: mergedHeaders,
	};
}

function toFlueTool(tool: McpServerTool): ToolDefinition {
	const { annotations } = tool.metadata.mcp;
	const definition: ToolDefinition = {
		name: tool.name,
		description: tool.description,
		input: undefined,
		output: undefined,
		...(annotations === undefined ? {} : { annotations }),
		run() {
			throw new Error('[flue] MCP tools execute through the internal adapter.');
		},
	};
	const execute = tool.execute;
	if (!execute) throw new Error(`[flue] MCP tool "${tool.name}" has no execute function.`);
	registerPreparedToolAdapter(definition, {
		parameters: tool.inputSchema ?? { type: 'object', properties: {} },
		async execute(args, signal) {
			if (signal?.aborted) throw new Error('Operation aborted');
			// ponytail: Flue has no custom-event stream for MCP calls; the MCP execute reads only the signal.
			return toToolContent(await execute(args, { abortSignal: signal, emitCustomEvent() {} }));
		},
	});
	return Object.freeze(definition);
}

function createToolName(serverName: string, toolName: string) {
	return `mcp__${sanitizeToolNamePart(serverName)}__${sanitizeToolNamePart(toolName)}`;
}

function sanitizeToolNamePart(value: string) {
	const sanitized = value.replace(/[^A-Za-z0-9_-]/g, '_').replace(/^_+|_+$/g, '');
	return sanitized || 'unnamed';
}

/** TanStack's MCP result (text, content parts, or structured content) as Flue tool content. */
function toToolContent(result: unknown): (TextContent | ImageContent)[] {
	if (typeof result === 'string') return [{ type: 'text', text: result }];
	if (Array.isArray(result)) return result.map((part: ContentPart) => toContentBlock(part));
	return [{ type: 'text', text: JSON.stringify(result, null, 2) }];
}

function toContentBlock(part: ContentPart): TextContent | ImageContent {
	if (part.type === 'text') return { type: 'text', text: part.content };
	if (part.type === 'image' && part.source.type === 'data') {
		return {
			type: 'image',
			data: part.source.value,
			mimeType: part.source.mimeType,
		};
	}
	return { type: 'text', text: JSON.stringify(part) };
}
