import { describe, expect, test } from "bun:test";
import { applyClaudeCodeCacheAlignment } from "../../../src/adapters/anthropic-cache-alignment";
import { createAnthropicAdapter } from "../../../src/adapters/anthropic";
import { parseRequest } from "../../../src/responses/parser";
import { CLAUDE_CODE_SYSTEM_INSTRUCTION } from "../../../src/oauth/anthropic";
import type { OcxProviderConfig } from "../../../src/types";

const cc = { type: "ephemeral" } as const;
const provider: OcxProviderConfig = {
  adapter: "anthropic", baseUrl: "https://gateway.invalid", authMode: "key",
  apiKey: "local-cache-test-key", models: ["claude-opus-4-8"],
};
const models = ["claude-opus-4-8", "claude-sonnet-5", "claude-fable-5"];
const tools = [
  { type: "function", name: "exec_command", parameters: {
    type: "object", properties: { cmd: { type: "string" } }, required: ["cmd"],
  } },
  { type: "custom", name: "apply_patch", format: { type: "text" } },
];
const input = [
  { role: "user", content: "First turn" },
  { role: "assistant", content: "Ready" },
  { role: "user", content: "Edit a file" },
];

function cachePaths(body: any): string[] {
  const paths: string[] = [];
  if (body.cache_control) paths.push("cache_control");
  for (const key of ["tools", "system"]) {
    body[key]?.forEach((block: any, i: number) => {
      if (block.cache_control) paths.push(`${key}[${i}]`);
    });
  }
  body.messages?.forEach((message: any, i: number) => {
    if (!Array.isArray(message.content)) return;
    message.content.forEach((block: any, j: number) => {
      if (block.cache_control) paths.push(`messages[${i}].content[${j}]`);
    });
  });
  return paths;
}

function payloadWithoutCache(body: any): any {
  const copy = structuredClone(body);
  delete copy.cache_control;
  for (const key of ["tools", "system"]) {
    for (const block of copy[key] ?? []) delete block.cache_control;
  }
  for (const message of copy.messages ?? []) {
    if (typeof message.content === "string") message.content = [{ type: "text", text: message.content }];
    for (const block of message.content ?? []) delete block.cache_control;
  }
  return copy;
}

async function build(
  alignment: boolean | undefined = true,
  overrides: Partial<OcxProviderConfig> = {},
  extra: Record<string, unknown> = {},
  retention?: "none" | "short" | "long",
) {
  const wire = await createAnthropicAdapter({
    ...provider, ...overrides,
    ...(alignment === undefined ? {} : { claudeCodeCacheAlignment: alignment }),
  }, retention).buildRequest(parseRequest({
    model: "claude-opus-4-8", instructions: "Keep the project conventions.",
    tools, input, stream: true, ...extra,
  }));
  return { wire, body: JSON.parse(wire.body) };
}

describe("captured Claude Code cache layout", () => {
  // Shape-only reconstruction of the direct CLI -> capture proxy -> sub2api
  // requests. Claude CLI 2.1.278 (external, sdk-cli), three explicit models.
  // Captures covered the initial request and retries, not interactive/multi-turn
  // behavior. Billing/prompt/tool text and request/account identifiers are omitted.
  test.each(models)("%s reproduces the observed three positions without adding text", model => {
    const body = {
      model,
      system: [
        { type: "text", text: "x-anthropic-billing-header: fixture" },
        { type: "text", text: "You are a Claude agent, built on Anthropic's Claude Agent SDK." },
        { type: "text", text: "Main prompt fixture" },
      ],
      tools: Array.from({ length: 20 }, (_, i) => ({ name: `tool_${i}`, input_schema: { type: "object" } })),
      messages: [
        { role: "user", content: [{ type: "text", text: "Probe" }] },
        { role: "system", content: [{ type: "text", text: "CLI tail fixture" }] },
      ],
    };
    const original = structuredClone(body);
    applyClaudeCodeCacheAlignment(body, cc);
    expect(cachePaths(body)).toEqual(["system[1]", "system[2]", "messages[1].content[0]"]);
    expect(payloadWithoutCache(body)).toEqual(original);
    const once = JSON.stringify(body);
    applyClaudeCodeCacheAlignment(body, cc);
    expect(JSON.stringify(body)).toBe(once);
  });

  test("relocates wire markers without touching fields inside schemas and tool data", () => {
    const data = { cache_control: { application: "not a wire marker" } };
    const body: any = {
      cache_control: cc,
      tools: [{ name: "task", input_schema: { type: "object", properties: data }, cache_control: cc }],
      system: ["old", "identity", "main"].map(text => ({ type: "text", text, cache_control: cc })),
      messages: [
        { role: "user", content: [{ type: "text", text: "previous", cache_control: cc }] },
        { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "task", input: data, cache_control: cc }] },
        { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: "done" }] },
      ],
    };
    const before = payloadWithoutCache(body);
    applyClaudeCodeCacheAlignment(body, cc);
    expect(cachePaths(body)).toEqual(["system[1]", "system[2]", "messages[2].content[0]"]);
    expect(payloadWithoutCache(body)).toEqual(before);
    expect(body.tools[0].input_schema.properties).toEqual(data);
    expect(body.messages[1].content[0].input).toEqual(data);
  });

  test.each(["image", "document", "tool_use", "tool_result"])("places the final marker on trailing %s, not earlier text", type => {
    const body = { messages: [{ role: "user", content: [{ type: "text", text: "prefix" }, { type }] }] };
    applyClaudeCodeCacheAlignment(body, cc);
    expect(cachePaths(body)).toEqual(["messages[0].content[1]"]);
  });

  test("never marks thinking, redacted thinking, or empty text and never creates content", () => {
    for (const content of [
      "", [], [{ type: "text", text: "" }],
      [{ type: "thinking", thinking: "private", signature: "test-signature" }],
      [{ type: "redacted_thinking", data: "test-data" }],
    ]) {
      const body = { messages: [{ role: "assistant", content }] };
      const before = structuredClone(body);
      applyClaudeCodeCacheAlignment(body, cc);
      expect(body).toEqual(before);
      expect(cachePaths(body)).toEqual([]);
    }
    const empty = {};
    applyClaudeCodeCacheAlignment(empty, cc);
    expect(empty).toEqual({});
  });
});

describe("opt-in Anthropic adapter integration", () => {
  test.each(models)("%s uses the same layout on native and gateway key endpoints", async model => {
    for (const baseUrl of ["https://api.anthropic.com", "https://gateway.invalid"]) {
      const { body } = await build(true, { baseUrl }, { model });
      expect(cachePaths(body)).toEqual(["system[0]", "messages[2].content[0]"]);
      expect(body.messages[0].content).toBe("First turn");
      expect(body.messages[2].content[0].cache_control).toEqual(cc);
      expect(body.system).toHaveLength(1);
      expect(body.system[0].text).not.toContain(CLAUDE_CODE_SYSTEM_INSTRUCTION);
    }
  });

  test("OAuth caches the existing SDK identity and main prompt without injecting another block", async () => {
    const { body, wire } = await build(true, { authMode: "oauth" });
    expect(body.system).toHaveLength(2);
    expect(body.system[0].text).toBe(CLAUDE_CODE_SYSTEM_INSTRUCTION);
    expect(cachePaths(body)).toEqual(["system[0]", "system[1]", "messages[2].content[0]"]);
    expect(wire.headers.Authorization).toBe("Bearer local-cache-test-key");
  });

  test("false and omitted retain upstream OpenCodex caching, including native automatic caching", async () => {
    for (const baseUrl of ["https://gateway.invalid", "https://api.anthropic.com"]) {
      const omitted = await createAnthropicAdapter({ ...provider, baseUrl }).buildRequest(parseRequest({
        model: "claude-opus-4-8", instructions: "Keep the project conventions.", tools, input, stream: true,
      }));
      const disabled = await build(false, { baseUrl });
      expect(disabled.wire).toEqual(omitted);
      expect(cachePaths(disabled.body)).toHaveLength(4);
      expect(disabled.body.tools.at(-1).cache_control).toEqual(cc);
      if (baseUrl === "https://api.anthropic.com") expect(disabled.body.cache_control).toEqual(cc);
    }
  });

  test("changes no non-cache payload fields, instructions, tool schemas, or headers", async () => {
    const aligned = await build(true);
    const upstream = await build(false);
    expect(payloadWithoutCache(aligned.body)).toEqual(payloadWithoutCache(upstream.body));
    expect(aligned.wire.headers).toEqual(upstream.wire.headers);
    expect(aligned.wire.url).toBe(upstream.wire.url);
    const patchTool = aligned.body.tools.find((tool: any) => tool.name === "apply_patch");
    expect(patchTool.input_schema.properties.input.type).toBe("string");
    expect(patchTool.input_schema.required).toEqual(["input"]);
  });

  test("multi-turn tool continuation moves the marker to the final tool result", async () => {
    const { body } = await build(true, {}, { input: [
      { role: "user", content: "Run a command" },
      { type: "function_call", call_id: "call_test", name: "exec_command", arguments: '{"cmd":"pwd"}' },
      { type: "function_call_output", call_id: "call_test", output: "/workspace" },
    ] });
    expect(cachePaths(body)).toEqual(["system[0]", "messages[2].content[0]"]);
    expect(body.messages[2].content[0].type).toBe("tool_result");
    expect(body.messages[2].content[0].tool_use_id).toBe(body.messages[1].content[0].id);
    expect(body.messages[2].content[0].content).toBe("/workspace");
  });

  test("missing system content stays absent rather than fabricating a third breakpoint", async () => {
    const { body } = await build(true, {}, { instructions: "", tools: [] });
    expect(body.system).toBeUndefined();
    expect(body.tools).toBeUndefined();
    expect(cachePaths(body)).toEqual(["messages[2].content[0]"]);
  });

  test("explicit retention still owns cache lifetime and opt-out", async () => {
    for (const retention of [undefined, "short", "long", "none"] as const) {
      const { body } = await build(true, {}, {}, retention);
      if (retention === "none") {
        expect(cachePaths(body)).toEqual([]);
      } else {
        const expected = retention === "long" ? { ...cc, ttl: "1h" } : cc;
        expect(body.system[0].cache_control).toEqual(expected);
        expect(body.messages[2].content[0].cache_control).toEqual(expected);
      }
    }
  });
});
