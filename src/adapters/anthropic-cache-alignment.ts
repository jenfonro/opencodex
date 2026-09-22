type Block = Record<string, unknown>;
type CacheControl = { type: "ephemeral"; ttl?: "1h" | "5m" };

const cacheableMessageTypes = new Set(["text", "image", "document", "tool_use", "tool_result"]);

/**
 * Opt-in layout based on Claude CLI 2.1.278 sdk-cli captures: two stable system
 * blocks (not the billing header), plus the final message block; no tool or
 * top-level cache marker. The adapter may have only one system block under key
 * auth. Do not invent identity/billing text to reach a fixed breakpoint count.
 */
export function applyClaudeCodeCacheAlignment(body: Block, cc: CacheControl | undefined): void {
  if (!cc) return;
  const tools = body.tools as Block[] | undefined;
  const system = body.system as Block[] | undefined;
  const messages = body.messages as Block[] | undefined;

  // Only wire-level markers belong to this policy. Never walk into tool schemas,
  // tool arguments, or nested tool-result data with a property named cache_control.
  delete body.cache_control;
  for (const blocks of [tools, system]) {
    for (const block of blocks ?? []) delete block.cache_control;
  }
  for (const message of messages ?? []) {
    if (Array.isArray(message.content)) {
      for (const block of message.content as Block[]) delete block.cache_control;
    }
  }

  const stableSystem = (system ?? []).filter(block =>
    block.type === "text"
    && typeof block.text === "string"
    && block.text.length > 0
    && !block.text.startsWith("x-anthropic-billing-header:"),
  );
  for (const block of stableSystem.slice(-2)) block.cache_control = { ...cc };

  const last = messages?.at(-1);
  if (!last) return;
  if (typeof last.content === "string") {
    if (last.content.length > 0) {
      last.content = [{ type: "text", text: last.content, cache_control: { ...cc } }];
    }
    return;
  }
  if (!Array.isArray(last.content)) return;
  for (let i = last.content.length - 1; i >= 0; i--) {
    const block = last.content[i] as Block;
    if (!cacheableMessageTypes.has(String(block.type))) continue;
    if (block.type === "text" && (typeof block.text !== "string" || block.text.length === 0)) continue;
    block.cache_control = { ...cc };
    break;
  }
}
