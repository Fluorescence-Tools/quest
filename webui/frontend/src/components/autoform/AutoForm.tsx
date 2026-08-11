"use client";

import React from "react";
import { AutoFormProps } from "./types";
import { AutoSection } from "./AutoSection";

export const AutoForm: React.FC<AutoFormProps> = ({
  spec,
  data,
  onChange,
  onSave,
  onAction,
  catalog,
}) => {
  if (!spec || !spec.sections) {
    return (
      <div className="autoform-empty" style={{ padding: "16px", color: "var(--text-muted)" }}>
        No AutoForm view spec loaded.
      </div>
    );
  }

  return (
    <div className="autoform-root" data-testid="autoform-root">
      {spec.title && (
        <h2 style={{ fontSize: "16px", marginBottom: "12px", borderBottom: "1px solid var(--line)", paddingBottom: "6px" }}>
          {spec.title}
        </h2>
      )}
      {spec.sections.map((section, idx) => (
        <AutoSection
          key={idx}
          section={section}
          data={data}
          onChange={onChange}
          onSave={onSave}
          onAction={onAction}
          catalog={catalog}
        />
      ))}
    </div>
  );
};
