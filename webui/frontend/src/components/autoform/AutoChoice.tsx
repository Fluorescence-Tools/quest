"use client";

import React from "react";
import { AutoFormSection } from "./types";
import HelpBadge from "../HelpBadge";


interface AutoChoiceProps {
  section: AutoFormSection;
  value: any;
  onChange: (val: any) => void;
}

export const AutoChoice: React.FC<AutoChoiceProps> = ({ section, value, onChange }) => {
  const label = section.label || section.title || section.attr || "";
  const helpText = section.description;
  const optionsRaw = section.options;

  let options: Array<{ label: string; value: any }> = [];
  if (Array.isArray(optionsRaw)) {
    options = optionsRaw.map((opt) =>
      typeof opt === "object" ? opt : { label: String(opt), value: opt }
    );
  } else if (optionsRaw && typeof optionsRaw === "object") {
    options = Object.entries(optionsRaw).map(([k, v]) => ({ label: k, value: v }));
  }

  return (
    <div className="field autoform-choice-field" style={{ marginBottom: "12px" }}>
      <label style={{ display: "block", marginBottom: "4px", fontSize: "13px" }}>
        <span className="help-badge-container" style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
          <strong>{label}</strong>
          {helpText && <HelpBadge definition={null} fallback={helpText} />}

        </span>
      </label>
      <select
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value)}
        style={{
          width: "100%",
          background: "var(--surface)",
          color: "var(--text)",
          border: "1px solid var(--line)",
          borderRadius: "4px",
          padding: "6px 8px",
        }}
      >
        {options.map((opt, i) => (
          <option key={i} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
};
