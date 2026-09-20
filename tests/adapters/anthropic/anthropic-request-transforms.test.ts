import { describe, expect, spyOn, test } from "bun:test";
import { createAnthropicAdapter, applyAnthropicPromptCachePolicy } from "../../../src/adapters/anthropic";
import { parseRequest } from "../../../src/responses/parser";
import { createTranslatorBudget } from "../../../src/lib/translator-budget";
import { buildResponseJSON, bridgeToResponsesSSE } from "../../../src/bridge";
import { getDefaultConfig, validateConfigCandidate, loadConfig, saveConfig } from "../../../src/config";
import { providerEditorConfigDTO, parseProviderEditorConfigDTO, providerManagementConfigError } from "../../../src/server/auth-cors";
import { handleManagementAPI } from "../../../src/server/management-api";
import * as destinationPolicy from "../../../src/lib/destination-policy";
import { catalogConvergenceFactory } from "../../helpers/catalog-convergence";
import { ManagementRequest } from "../../helpers/management-auth";
import { repoRoot } from "../../helpers/repo-root";

const off = { promptCaching: false, identityRewrite: false, toolCatalogNudge: false };
const provider = (policy?: Record<string, unknown>, baseUrl = 'https://gateway.invalid') => ({
  adapter: 'anthropic', baseUrl, authMode: 'key', apiKey: 'local-test-not-a-secret',
  models: ['claude-opus-5'],
  ...(policy === undefined ? {} : { anthropicRequestTransforms: policy }),
});
const instructions = 'You are Codex, an agent based on GPT-6.\nKeep this exact caller text. 中文';
const patch = '*** Begin Patch\n*** Add File: probe.txt\n+native-tool-ok\n*** End Patch';
const tools = [
  { type: 'function', name: 'exec_command', description: 'Run a command', parameters: {
    type: 'object', properties: { cmd: { type: 'string' } }, required: ['cmd'],
  } },
  { type: 'custom', name: 'apply_patch', description: 'Apply a patch', format: { type: 'text' } },
  { type: 'tool_search' },
];
const request = (extra = {}) => ({
  model: 'claude-opus-5', instructions, tools, stream: true,
  input: [{ role: 'user', content: 'First turn' }, { role: 'assistant', content: 'Ready' }, { role: 'user', content: 'Edit a file' }],
  ...extra,
});
async function build(policy?: Record<string, unknown>, extra = {}, baseUrl?: string, retention?: 'none' | 'short' | 'long') {
  const result = await createAnthropicAdapter(provider(policy, baseUrl), retention).buildRequest(parseRequest(request(extra)));
  return { wire: result, body: JSON.parse(result.body) };
}
function markers(value: unknown): unknown[] {
  if (!value || typeof value !== 'object') return [];
  return Object.entries(value).flatMap(([key, child]) => key === 'cache_control' ? [child] : markers(child));
}
function freezeDeep(value: any): any {
  if (value && typeof value === 'object') {
    Object.values(value).forEach(freezeDeep);
    Object.freeze(value);
  }
  return value;
}

describe('provider-scoped optional mutations', () => {
  test('minimal policy preserves caller instructions and native tool declarations', async () => {
    const { wire, body } = await build(off);
    expect(body.system).toEqual([{ type: 'text', text: instructions }]);
    expect(markers(body)).toEqual([]);
    expect(body.tools.map((t: any) => t.name)).toEqual(['exec_command', 'apply_patch', 'tool_search']);
    expect(body.tools[0].input_schema).toEqual(tools[0].parameters);
    expect(body.tools[1].input_schema.properties.input.type).toBe('string');
    expect(body.tools[1].input_schema.required).toEqual(['input']);
    expect(wire.url).toBe('https://gateway.invalid/v1/messages');
    expect(wire.headers['x-api-key']).toBe('local-test-not-a-secret');
  });

  test('omitted flags keep upstream defaults', async () => {
    const original = (await build()).body;
    expect((await build({})).body).toEqual(original);
    expect((await build({ promptCaching: true, identityRewrite: true, toolCatalogNudge: true })).body).toEqual(original);
    expect(markers(original).length).toBe(4);
    expect(original.system[0].text).toContain('powered by the claude-opus-5');
    expect(original.system[0].text).toContain('Tool contract:');
  });

  test('each switch is independent', async () => {
    const cacheOff = (await build({ promptCaching: false })).body;
    expect(markers(cacheOff)).toEqual([]);
    expect(cacheOff.system[0].text).toContain('powered by the claude-opus-5');
    expect(cacheOff.system[0].text).toContain('Tool contract:');
    const identityOff = (await build({ identityRewrite: false })).body;
    expect(identityOff.system[0].text).toStartWith(instructions);
    expect(identityOff.system[0].text).toContain('Tool contract:');
    expect(markers(identityOff).length).toBe(4);
    const nudgeOff = (await build({ toolCatalogNudge: false })).body;
    expect(nudgeOff.system[0].text).not.toContain('Tool contract:');
    expect(nudgeOff.system[0].text).toContain('powered by the claude-opus-5');
    expect(markers(nudgeOff).length).toBe(4);
  });

  test('cache opt-out wins over every retention and native automatic caching', async () => {
    for (const baseUrl of ['https://gateway.invalid', 'https://api.anthropic.com']) {
      for (const retention of [undefined, 'none', 'short', 'long'] as const) {
        expect(markers((await build(off, {}, baseUrl, retention)).body)).toEqual([]);
      }
    }
    expect((await build(undefined, {}, 'https://api.anthropic.com')).body.cache_control).toEqual({ type: 'ephemeral' });
  });

  test('cache opt-out neither trims existing markers nor normalizes TTLs', () => {
    const body = freezeDeep({
      cache_control: { type: 'ephemeral', ttl: '1h' },
      tools: [{ name: 'a', cache_control: { type: 'ephemeral', ttl: '5m' } }],
      system: [{ type: 'text', text: 'caller', cache_control: { type: 'ephemeral', ttl: '1h' } }],
      messages: Array.from({ length: 5 }, (_, i) => ({ role: 'user', content: [
        { type: 'text', text: String(i), cache_control: { type: 'ephemeral', ttl: '1h' } },
      ] })),
    });
    const before = JSON.stringify(body);
    for (const retention of [undefined, 'none', 'short', 'long'] as const) {
      applyAnthropicPromptCachePolicy(body, provider(off), retention);
      expect(JSON.stringify(body)).toBe(before);
    }
    const legacy = structuredClone(body);
    delete legacy.cache_control;
    applyAnthropicPromptCachePolicy(legacy, provider(), 'none');
    expect(markers(legacy).length).toBe(4);
    expect(legacy.system[0].cache_control).toEqual({ type: 'ephemeral' });
  });

  test('empty caller prompt stays absent without the extra tool prompt', async () => {
    expect((await build(off, { instructions: '' })).body.system).toBeUndefined();
    expect((await build(off, { tools: [] })).body.tools).toBeUndefined();
  });

  test('tool choice and authentication conversion remain intact', async () => {
    const body = (await build(off, { tool_choice: { type: 'custom', name: 'apply_patch' } })).body;
    expect(body.tool_choice).toEqual({ type: 'tool', name: 'apply_patch' });
    const oauth = { ...provider(off, 'https://api.anthropic.com'), authMode: 'oauth' };
    const wire = await createAnthropicAdapter(oauth).buildRequest(parseRequest(request()));
    const oauthBody = JSON.parse(wire.body);
    expect(wire.headers.Authorization).toBe('Bearer local-test-not-a-secret');
    expect(oauthBody.system[0].text).toContain('Claude');
    expect(oauthBody.system[1].text).toBe(instructions);
    expect(markers(oauthBody)).toEqual([]);
  });
});

describe('native custom-tool round trip', () => {
  const signature = 'AbCdEf0123456789AbCdEf0123456789==';
  const upstream = {
    id: 'msg_probe', type: 'message', role: 'assistant', model: 'claude-opus-5',
    content: [
      { type: 'thinking', thinking: 'Test reasoning', signature },
      { type: 'tool_use', id: 'toolu_probe', name: 'apply_patch', input: { input: patch } },
    ], stop_reason: 'tool_use', usage: { input_tokens: 10, output_tokens: 20 },
  };
  test('Claude tool_use becomes custom_tool_call; patch input, signature and result pairing survive replay', async () => {
    const adapter = createAnthropicAdapter(provider(off));
    const budget = createTranslatorBudget();
    try {
      const events = await adapter.parseResponse(Response.json(upstream), budget);
      expect(events).toContainEqual({ type: 'thinking_signature', signature });
      const response = buildResponseJSON(events, 'claude-opus-5', { freeformToolNames: new Set(['apply_patch']) });
      const call = response.output.find((item: any) => item.type === 'custom_tool_call');
      expect(call.name).toBe('apply_patch');
      expect(call.input).toBe(patch);
      const replay = await adapter.buildRequest(parseRequest(request({ input: [
        { role: 'user', content: 'Edit a file' }, ...response.output,
        { type: 'custom_tool_call_output', call_id: call.call_id, output: 'Success. Added probe.txt' },
      ] })));
      const messages = JSON.parse(replay.body).messages;
      const assistant = messages.find((m: any) => m.role === 'assistant');
      expect(assistant.content).toContainEqual({ type: 'thinking', thinking: 'Test reasoning', signature });
      const toolUse = assistant.content.find((b: any) => b.type === 'tool_use');
      expect(toolUse.input).toEqual({ input: patch });
      const result = messages.flatMap((m: any) => Array.isArray(m.content) ? m.content : []).find((b: any) => b.type === 'tool_result');
      expect(result.tool_use_id).toBe(toolUse.id);
      expect(JSON.stringify(result.content)).toContain('Success. Added probe.txt');
    } finally { budget.dispose(); }
  });

  test('streamed patch arguments round-trip into native custom-tool SSE', async () => {
    const json = JSON.stringify({ input: patch });
    const chunks = [
      { type: 'message_start', message: { ...upstream, content: [] } },
      { type: 'content_block_start', index: 0, content_block: { type: 'tool_use', id: 'toolu_stream', name: 'apply_patch', input: {} } },
      ...[json.slice(0, 12), json.slice(12)].map(partial_json => ({ type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json } })),
      { type: 'content_block_stop', index: 0 },
      { type: 'message_delta', delta: { stop_reason: 'tool_use' }, usage: { output_tokens: 20 } },
      { type: 'message_stop' },
    ];
    const wire = chunks.map(chunk => 'event: ' + chunk.type + '\ndata: ' + JSON.stringify(chunk) + '\n\n').join('');
    const budget = createTranslatorBudget();
    try {
      const events = createAnthropicAdapter(provider(off)).parseStream(new Response(wire), budget);
      const stream = bridgeToResponsesSSE(events, 'claude-opus-5', undefined, new Set(['apply_patch']));
      const text = await new Response(stream).text();
      const frames = text.split('\n').filter(line => line.startsWith('data: {')).map(line => JSON.parse(line.slice(6)));
      // The current bridge buffers apply_patch until its complete input is validated.
      expect(frames.find(frame => frame.type === 'response.custom_tool_call_input.done').input).toBe(patch);
      const completed = frames.find(frame => frame.type === 'response.completed');
      expect(completed.response.output.find((item: any) => item.type === 'custom_tool_call').input).toBe(patch);
    } finally { budget.dispose(); }
  });
});

describe('config persistence and validation', () => {
  const config = (policy: any) => ({ ...getDefaultConfig(), defaultProvider: 'claude', providers: { claude: provider(policy) } });
  test("management policy validation initializes in a fresh process before config", () => {
    const child = Bun.spawnSync([
      process.execPath, "--eval", `
        const { providerManagementConfigError } = await import("./src/server/auth-cors.ts");
        const { getDefaultConfig } = await import("./src/config.ts");
        const provider = {
          adapter: "anthropic", baseUrl: "https://gateway.invalid", authMode: "key",
          apiKey: "fixture-key", models: ["claude-opus-5"],
          anthropicRequestTransforms: { promptCaching: "false" },
        };
        const error = providerManagementConfigError("claude", provider);
        if (!error?.includes("anthropicRequestTransforms")) throw new Error(String(error));
        if (getDefaultConfig().port !== 10100) throw new Error("config did not initialize");
        console.log("policy and config initialized");
      `,
    ], { cwd: repoRoot(), env: process.env, timeout: 20_000 });
    expect(new TextDecoder().decode(child.stderr)).toBe("");
    expect(child.exitCode).toBe(0);
    expect(new TextDecoder().decode(child.stdout).trim()).toBe("policy and config initialized");
  }, 25_000);

  test('policy survives validation and the management editor without exposing secrets', () => {
    const result = validateConfigCandidate(config(off));
    expect(result.ok).toBe(true);
    expect(result.config.providers.claude.anthropicRequestTransforms).toEqual(off);
    saveConfig(result.config);
    expect(loadConfig().providers.claude.anthropicRequestTransforms).toEqual(off);
    const dto = providerEditorConfigDTO(result.config);
    expect(dto.providers.claude.anthropicRequestTransforms).toEqual(off);
    expect(dto.providers.claude.apiKey).toBeUndefined();
    expect(parseProviderEditorConfigDTO(dto).ok).toBe(true);
  });
  test('wrong types and misspelled policy fields fail validation', () => {
    for (const key of Object.keys(off)) {
      for (const value of ['false', 0, null, []]) {
        expect(validateConfigCandidate(config({ [key]: value })).ok).toBe(false);
        expect(providerManagementConfigError("claude", provider({ [key]: value }))).toContain("anthropicRequestTransforms");
      }
    }
    expect(validateConfigCandidate(config({ promptCache: false })).ok).toBe(false);
    expect(validateConfigCandidate(config(null)).ok).toBe(false);
  });

  test('management validation never echoes invalid keys or values', () => {
    const secret = 'test-private-policy-value';
    const error = providerManagementConfigError('claude', provider({ [secret]: secret }));
    expect(error).toContain('anthropicRequestTransforms');
    expect(error).not.toContain(secret);
  });

  test('an unrelated provider form save preserves the policy; explicit replacement wins', async () => {
    const liveConfig = config(off);
    saveConfig(liveConfig);
    const destination = spyOn(destinationPolicy, 'providerDestinationResolvedError').mockResolvedValue(null);
    const post = async (submitted: ReturnType<typeof provider>) => {
      const req = new ManagementRequest('http://127.0.0.1/api/providers', {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ name: 'claude', provider: submitted }),
      });
      return handleManagementAPI(req, new URL(req.url), liveConfig, {
        createManagementConvergeCodex: catalogConvergenceFactory(() => {}),
      });
    };
    try {
      expect((await post(provider()))?.status).toBe(200);
      expect(loadConfig().providers.claude.anthropicRequestTransforms).toEqual(off);
      expect(liveConfig.providers.claude.anthropicRequestTransforms).toEqual(off);
      const replacement = { promptCaching: false };
      expect((await post(provider(replacement)))?.status).toBe(200);
      expect(loadConfig().providers.claude.anthropicRequestTransforms).toEqual(replacement);
      expect((await post(provider({})))?.status).toBe(200);
      expect(loadConfig().providers.claude.anthropicRequestTransforms).toEqual({});
    } finally { destination.mockRestore(); }
  });
});
