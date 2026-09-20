import * as z from "zod/v4";

// Keep this policy independent of the config facade: management validation can
// load first, and importing the aggregate schema there creates an init cycle.
export const anthropicRequestTransformsSchema = z.object({
  promptCaching: z.boolean().optional(),
  identityRewrite: z.boolean().optional(),
  toolCatalogNudge: z.boolean().optional(),
}).strict();

export function anthropicRequestTransformsConfigError(value: unknown): string | null {
  if (value === undefined || anthropicRequestTransformsSchema.safeParse(value).success) return null;
  // Supplied values and unknown keys can contain secrets; never echo them.
  return "anthropicRequestTransforms must contain only optional boolean promptCaching, identityRewrite, and toolCatalogNudge fields";
}
