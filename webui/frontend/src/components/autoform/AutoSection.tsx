"use client";

import React from "react";
import { AutoFormSection } from "./types";
import { AutoPanel } from "./AutoPanel";
import { AutoValue } from "./AutoValue";
import { AutoChoice } from "./AutoChoice";
import { AutoToggle } from "./AutoToggle";
import { getNestedValue } from "./utils";
import { attrToPath } from "../../lib/autoformPaths";

interface AutoSectionProps {
  section: AutoFormSection;
  data: Record<string, any>;
  onChange: (path: string, val: any) => void;
  onSave?: (path: string) => void;
  onAction?: (actionName: string, payload?: any) => void;
  catalog?: Record<string, any>;
}

export const AutoSection: React.FC<AutoSectionProps> = ({
  section,
  data,
  onChange,
  onSave,
  onAction,
  catalog,
}) => {
  const path = attrToPath(section.attr || section.target || "");

  switch (section.type) {
    case "panel":
    case "dock_area":
      if (section.sections) {
        return (
          <AutoPanel
            section={section}
            data={data}
            onChange={onChange}
            onSave={onSave}
            onAction={onAction}
            catalog={catalog}
          />
        );
      }
      return null;

    case "value":
      return (
        <AutoValue
          section={section}
          value={path ? getNestedValue(data, path) : undefined}
          onChange={(val) => path && onChange(path, val)}
          onSave={() => path && onSave?.(path)}
        />
      );

    case "choice":
      return (
        <AutoChoice
          section={section}
          value={path ? getNestedValue(data, path) : undefined}
          onChange={(val) => path && onChange(path, val)}
        />
      );

    case "toggle":
    case "toggle_row":
      return (
        <AutoToggle
          section={section}
          value={path ? getNestedValue(data, path) : undefined}
          onChange={(val) => path && onChange(path, val)}
        />
      );

    case "button_row":
      return (
        <div className="field autoform-button-row" style={{ display: "flex", gap: "8px", marginBottom: "12px" }}>
          {section.sections?.map((b, i) => (
            <button
              key={i}
              type="button"
              onClick={() => onAction?.(b.attr || b.label || "click")}
              style={{
                background: "var(--accent)",
                color: "#fff",
                border: "none",
                borderRadius: "4px",
                padding: "6px 12px",
                cursor: "pointer",
              }}
            >
              {b.label || b.title}
            </button>
          ))}
        </div>
      );

    case "info":
      return (
        <div
          className="autoform-info"
          style={{
            padding: "10px 12px",
            background: "var(--surface-2)",
            borderRadius: "4px",
            marginBottom: "12px",
            fontSize: "13px",
            lineHeight: 1.5,
          }}
        >
          {section.description || section.label || section.title}
        </div>
      );

    case "custom":
      if (section.options?.text) {
        return (
          <div
            className="autoform-custom-text"
            style={{
              padding: "12px",
              background: "var(--surface-2)",
              borderRadius: "6px",
              marginBottom: "12px",
              fontSize: "13px",
              whiteSpace: "pre-wrap",
            }}
          >
            {section.options.text}
          </div>
        );
      }
      return null;

    default:
      if (section.sections) {
        return (
          <div className="autoform-group">
            {section.sections.map((child, idx) => (
              <AutoSection
                key={idx}
                section={child}
                data={data}
                onChange={onChange}
                onSave={onSave}
                onAction={onAction}
                catalog={catalog}
              />
            ))}
          </div>
        );
      }
      return null;
  }
};
