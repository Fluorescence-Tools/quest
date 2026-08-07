"use client";

import React, { useState } from "react";
import { AutoFormSection } from "./types";
import { AutoSection } from "./AutoSection";

interface AutoPanelProps {
  section: AutoFormSection;
  data: Record<string, any>;
  onChange: (path: string, val: any) => void;
  onSave?: (path: string) => void;
  onAction?: (actionName: string, payload?: any) => void;
  catalog?: Record<string, any>;
}

export const AutoPanel: React.FC<AutoPanelProps> = ({
  section,
  data,
  onChange,
  onSave,
  onAction,
  catalog,
}) => {
  const [collapsed, setCollapsed] = useState(section.collapsed ?? false);
  const title = section.title || section.label || "Panel";

  return (
    <section className="panel-section autoform-panel" style={{ border: "1px solid var(--line)", borderRadius: "6px", marginBottom: "12px", background: "var(--surface)" }}>
      <div
        className="panel-header"
        onClick={() => setCollapsed(!collapsed)}
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "10px 14px",
          cursor: "pointer",
          background: "var(--surface-2)",
          borderBottom: collapsed ? "none" : "1px solid var(--line)",
          borderTopLeftRadius: "5px",
          borderTopRightRadius: "5px",
        }}
      >
        <h3 style={{ margin: 0, fontSize: "14px", fontWeight: 600 }}>{title}</h3>
        <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>{collapsed ? "▶" : "▼"}</span>
      </div>
      {!collapsed && (
        <div className="panel-body" style={{ padding: "12px 14px" }}>
          {section.sections?.map((childSec, idx) => (
            <AutoSection
              key={idx}
              section={childSec}
              data={data}
              onChange={onChange}
              onSave={onSave}
              onAction={onAction}
              catalog={catalog}
            />
          ))}
        </div>
      )}
    </section>
  );
};
