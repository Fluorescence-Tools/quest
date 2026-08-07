"use client";

import React from "react";
import { AutoFormSection } from "./types";
import HelpBadge from "../HelpBadge";


interface AutoToggleProps {
  section: AutoFormSection;
  value: any;
  onChange: (val: boolean) => void;
}

export const AutoToggle: React.FC<AutoToggleProps> = ({ section, value, onChange }) => {
  const label = section.label || section.title || section.attr || "";
  const helpText = section.description;

  return (
    <div className="field autoform-toggle-field" style={{ marginBottom: "12px" }}>
      <label className="check-field" style={{ display: "inline-flex", alignItems: "center", gap: "8px", cursor: "pointer" }}>
        <input
          type="checkbox"
          checked={Boolean(value)}
          onChange={(e) => onChange(e.target.checked)}
        />
        <span className="help-badge-container" style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
          <strong>{label}</strong>
          {helpText && <HelpBadge definition={null} fallback={helpText} />}

        </span>
      </label>
    </div>
  );
};
