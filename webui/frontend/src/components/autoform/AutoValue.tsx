"use client";

import React from "react";
import { AutoFormSection } from "./types";
import HelpBadge from "../HelpBadge";


interface AutoValueProps {
  section: AutoFormSection;
  value: any;
  onChange: (val: any) => void;
  onSave?: () => void;
}

export const AutoValue: React.FC<AutoValueProps> = ({ section, value, onChange, onSave }) => {
  const label = section.label || section.title || section.attr || "";
  const kind = section.kind || "str";
  const helpText = section.description;

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    let raw = e.target.value;
    if (kind === "int") {
      const parsed = parseInt(raw, 10);
      onChange(isNaN(parsed) ? 0 : parsed);
    } else if (kind === "float") {
      const parsed = parseFloat(raw);
      onChange(isNaN(parsed) ? 0 : parsed);
    } else {
      onChange(raw);
    }
  };

  return (
    <div className="field autoform-value-field" style={{ marginBottom: "12px" }}>
      <label style={{ display: "block", marginBottom: "4px", fontSize: "13px" }}>
        <span className="help-badge-container" style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
          <strong>{label}</strong>
          {helpText && <HelpBadge definition={null} fallback={helpText} />}

        </span>
      </label>
      {kind === "text" ? (
        <textarea
          value={value ?? ""}
          onChange={handleChange}
          style={{
            width: "100%",
            minHeight: "70px",
            background: "var(--surface)",
            color: "var(--text)",
            border: "1px solid var(--line)",
            borderRadius: "4px",
            padding: "6px 8px",
          }}
        />
      ) : (
        <input
          type={kind === "int" || kind === "float" ? "number" : "text"}
          step={kind === "float" ? (section.step || "any") : 1}
          min={section.minimum}
          max={section.maximum}
          value={value ?? ""}
          onChange={handleChange}
          style={{
            width: "100%",
            background: "var(--surface)",
            color: "var(--text)",
            border: "1px solid var(--line)",
            borderRadius: "4px",
            padding: "6px 8px",
          }}
        />
      )}
    </div>
  );
};
