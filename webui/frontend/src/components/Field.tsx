"use client";

import React from "react";
import { ParameterCatalog, fieldLabel, getParameterHelp } from "../lib/parameterCatalog";
import HelpBadge from "./HelpBadge";

interface FieldProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  step?: string;
  min?: number;
  max?: number;
  parameterId?: string;
  catalog?: ParameterCatalog | null;
  onSave?: () => void;
  isDirty?: boolean;
}

export default function Field({
  label,
  value,
  onChange,
  step = "0.1",
  min,
  max,
  parameterId,
  catalog,
  onSave,
  isDirty = false,
}: FieldProps) {
  const definition = parameterId && catalog ? getParameterHelp(catalog, parameterId) : null;
  return (
    <label className="field">
      <span className="help-badge-container">
        {fieldLabel(catalog, parameterId, label)}
        {parameterId && catalog && <HelpBadge definition={definition} />}
      </span>
      <div style={{ display: "flex", gap: "4px", alignItems: "center", width: "100%" }}>
        <input
          type="number"
          step={step}
          min={min}
          max={max}
          value={value}
          onChange={(event) => onChange(Number(event.target.value))}
          style={{ flexGrow: 1 }}
        />
        {onSave && (
          <button
            type="button"
            onClick={onSave}
            disabled={!isDirty}
            title="Save parameter: clicking makes persistent across sessions"
            style={{
              padding: "3px 6px",
              fontSize: "12px",
              minHeight: "unset",
              height: "26px",
              display: "flex",
              alignItems: "center",
              cursor: isDirty ? "pointer" : "not-allowed",
              opacity: isDirty ? 1 : 0.4,
              backgroundColor: isDirty ? undefined : "#f0f0f0",
              color: isDirty ? undefined : "#888",
              border: isDirty ? undefined : "1px solid #ccc",
              transition: "all 0.2s ease",
            }}
          >
            💾
          </button>
        )}
      </div>
    </label>
  );
}
