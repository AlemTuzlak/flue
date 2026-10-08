import { InMemoryTransport, Server, type Tool } from '@modelcontextprotocol/server';
import { describe, expect, it } from 'vitest';
import { createMcpConnectionFromTransport } from './mcp.ts';
import { assertToolDefinition, defineTool } from './tool.ts';

/**
 * Connects to an in-memory MCP server that lists `tools`. The tools are only
 * listed in these tests, never called.
 */
async function connectTo(name: string, tools: Tool[]) {
	const server = new Server({ name, version: '1.0.0' }, { capabilities: { tools: {} } });
	server.setRequestHandler('tools/list', () => ({ tools }));
	const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
	await server.connect(serverTransport);
	return createMcpConnectionFromTransport(name, clientTransport);
}

describe('MCP tool annotations', () => {
	it("carries the server's annotations through to the adapted tool definition", async () => {
		const connection = await connectTo('test', [
			{
				name: 'create_issue',
				title: 'Create Issue',
				description: 'Creates a new issue.',
				inputSchema: { type: 'object', properties: {}, required: [] },
				annotations: {
					title: 'Create Issue',
					readOnlyHint: false,
					destructiveHint: true,
					idempotentHint: false,
					openWorldHint: false,
				},
			},
		]);

		expect(connection.tools).toHaveLength(1);
		const tool = connection.tools[0];
		if (!tool) throw new Error('Expected one adapted MCP tool.');
		expect(tool.name).toBe('mcp__test__create_issue');
		expect(tool.annotations).toEqual({
			title: 'Create Issue',
			readOnlyHint: false,
			destructiveHint: true,
			idempotentHint: false,
			openWorldHint: false,
		});
		expect(Object.isFrozen(tool.annotations)).toBe(true);
		expect(Object.isFrozen(tool)).toBe(true);
		expect(tool.description).toBe('Creates a new issue.');
	});

	it('omits annotations when the server declares none', async () => {
		const connection = await connectTo('test', [
			{
				name: 'search_issues',
				description: 'Searches issues.',
				inputSchema: { type: 'object', properties: {}, required: [] },
			},
		]);

		expect(connection.tools[0]?.annotations).toBeUndefined();
		expect(connection.tools[0]?.name).toBe('mcp__test__search_issues');
	});

	it('accepts annotations on hand-written tool definitions', () => {
		const tool = defineTool({
			name: 'wipe_data',
			description: 'Deletes everything.',
			annotations: { destructiveHint: true },
			run: () => ({ output: 'wiped' }),
		});
		expect(tool.annotations).toEqual({ destructiveHint: true });
		expect(Object.isFrozen(tool.annotations)).toBe(true);
		// The same validation path useTool() runs accepts the field.
		expect(() => assertToolDefinition(tool, 'test')).not.toThrow();
	});

	it('rejects malformed annotations in the definition validation', () => {
		expect(() =>
			assertToolDefinition(
				{
					name: 'wipe_data',
					description: 'Deletes everything.',
					annotations: { destructiveHint: 'yes' },
					run: () => undefined,
				},
				'test',
			),
		).toThrow(/annotations\.destructiveHint must be a boolean/);

		expect(() =>
			assertToolDefinition(
				{
					name: 'wipe_data',
					description: 'Deletes everything.',
					annotations: { readOnlyhint: true },
					run: () => undefined,
				},
				'test',
			),
		).toThrow(/annotations received unknown field "readOnlyhint"/);
	});
});
