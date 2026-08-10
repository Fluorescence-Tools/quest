export interface ParameterDefinition {
  label: string;
  category: string;
  type: string;
  unit?: string;
  description: string;
}

export interface ParameterCatalog {
  version: number;
  title: string;
  description: string;
  documentation_path: string;
  /** Language the backend translated this catalog into ("en", "de", "fr"). */
  locale?: string;
  parameters: Record<string, ParameterDefinition>;
}

/**
 * The label to show for a field.
 *
 * In English the hand-written labels win: they are terser than the catalog's
 * prose ("t max [ns]" rather than "Simulation time") and were chosen for this
 * layout. In any other language there is no hand-written label to prefer, so
 * the catalog's translation is used — the same string the desktop form and
 * `doc/parameters.<code>.md` show.
 */
export function fieldLabel(
  catalog: ParameterCatalog | null | undefined,
  key: string | undefined,
  fallback: string
): string {
  if (!catalog || !key || !catalog.locale || catalog.locale === "en") return fallback;
  return catalog.parameters?.[key]?.label ?? fallback;
}

export function getParameterHelp(
  catalog: ParameterCatalog | null,
  key: string
): ParameterDefinition | null {
  return catalog?.parameters?.[key] ?? null;
}

export function formatHelpTitle(definition: ParameterDefinition | null): string | undefined {
  if (!definition) return undefined;
  const unit = definition.unit ? ` Unit: ${definition.unit}.` : "";
  return `${definition.label}: ${definition.description}${unit}`;
}
