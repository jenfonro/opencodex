import { describe, expect, spyOn, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { getDefaultConfig, loadConfig, saveConfig, validateConfigCandidate } from "../../src/config";
import * as destinationPolicy from "../../src/lib/destination-policy";
import { parseProviderEditorConfigDTO, providerEditorConfigDTO, providerManagementConfigError } from "../../src/server/auth-cors";
import { handleManagementAPI } from "../../src/server/management-api";
import type { OcxProviderConfig } from "../../src/types";
import { catalogConvergenceFactory } from "../helpers/catalog-convergence";
import { ManagementRequest } from "../helpers/management-auth";
import { removeTreeWithRetry } from "../helpers/remove-tree";

const provider: OcxProviderConfig = {
  adapter: "anthropic", baseUrl: "https://relay.example.test", authMode: "key",
  apiKey: "local-cache-config-test-key", liveModels: false, models: ["claude-opus-4-8"],
};
const config = (value?: unknown) => ({
  ...getDefaultConfig(),
  defaultProvider: "relay",
  providers: { relay: {
    ...provider,
    ...(value === undefined ? {} : { claudeCodeCacheAlignment: value }),
  } },
});

describe("Claude Code cache alignment provider config", () => {
  test("accepts only optional booleans, including an explicit false", () => {
    for (const value of [undefined, false, true]) {
      const result = validateConfigCandidate(config(value));
      expect(result.ok).toBe(true);
      expect(result.config.providers.relay.claudeCodeCacheAlignment).toBe(value);
    }
    for (const value of ["false", 0, 1, null, [], {}]) {
      expect(validateConfigCandidate(config(value)).ok).toBe(false);
      const error = providerManagementConfigError("relay", config(value).providers.relay);
      expect(error).toBe("provider relay claudeCodeCacheAlignment must be a boolean");
    }
  });

  test("invalid values are not echoed by management validation", () => {
    const privateValue = "fixture-private-invalid-setting";
    const error = providerManagementConfigError("relay", config(privateValue).providers.relay);
    expect(error).toContain("claudeCodeCacheAlignment");
    expect(error).not.toContain(privateValue);
  });

  test("persists through the editor, unrelated form saves, and explicit changes", async () => {
    const testDir = mkdtempSync(join(tmpdir(), "ocx-cache-alignment-config-"));
    const previousOcxHome = process.env.OPENCODEX_HOME;
    process.env.OPENCODEX_HOME = testDir;
    const live = { ...getDefaultConfig(), defaultProvider: "relay", providers: {
      relay: { ...provider, claudeCodeCacheAlignment: true },
    } };
    const resolved = spyOn(destinationPolicy, "providerDestinationResolvedError").mockResolvedValue(null);
    const request = async (method: string, path: string, body: unknown) => {
      const url = new URL(`http://localhost${path}`);
      return (await handleManagementAPI(new ManagementRequest(url, {
        method, headers: { "content-type": "application/json" }, body: JSON.stringify(body),
      }), url, live, { createManagementConvergeCodex: catalogConvergenceFactory() }))!;
    };
    try {
      saveConfig(live);
      expect(loadConfig().providers.relay.claudeCodeCacheAlignment).toBe(true);
      const editor = providerEditorConfigDTO(live);
      expect(editor.providers.relay.claudeCodeCacheAlignment).toBe(true);
      expect(editor.providers.relay.apiKey).toBeUndefined();
      expect(parseProviderEditorConfigDTO(editor).ok).toBe(true);

      for (const explicit of [undefined, false, undefined, true]) {
        const submitted = { ...provider, ...(explicit === undefined ? {} : { claudeCodeCacheAlignment: explicit }) };
        const expected = explicit ?? live.providers.relay.claudeCodeCacheAlignment;
        expect((await request("POST", "/api/providers", { name: "relay", provider: submitted })).status).toBe(200);
        expect(live.providers.relay.claudeCodeCacheAlignment).toBe(expected);
        expect(loadConfig().providers.relay.claudeCodeCacheAlignment).toBe(expected);
      }

      expect((await request("PATCH", "/api/providers?name=relay", { claudeCodeCacheAlignment: "false" })).status).toBe(400);
      expect(loadConfig().providers.relay.claudeCodeCacheAlignment).toBe(true);
      expect((await request("PATCH", "/api/providers?name=relay", { claudeCodeCacheAlignment: false })).status).toBe(200);
      expect(loadConfig().providers.relay.claudeCodeCacheAlignment).toBe(false);

      const baseline = providerEditorConfigDTO(loadConfig());
      const next = structuredClone(baseline);
      next.providers.relay.claudeCodeCacheAlignment = true;
      expect((await request("PUT", "/api/providers", { baseline, next })).status).toBe(200);
      expect(loadConfig().providers.relay.claudeCodeCacheAlignment).toBe(true);
    } finally {
      resolved.mockRestore();
      if (previousOcxHome === undefined) delete process.env.OPENCODEX_HOME;
      else process.env.OPENCODEX_HOME = previousOcxHome;
      removeTreeWithRetry(testDir);
    }
  });
});
