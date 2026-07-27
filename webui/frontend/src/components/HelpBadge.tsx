"use client";

import React, { useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { formatHelpTitle, ParameterDefinition } from "../lib/parameterCatalog";

interface HelpBadgeProps {
  definition: ParameterDefinition | null;
  fallback?: string;
}

export default function HelpBadge({ definition, fallback }: HelpBadgeProps) {
  const title = formatHelpTitle(definition) ?? fallback;
  const tooltipId = useId();
  const badgeRef = useRef<HTMLSpanElement>(null);
  const [isOpen, setIsOpen] = useState(false);
  const [position, setPosition] = useState({ left: 0, top: 0 });

  if (!title) return null;

  const openTooltip = () => {
    const rect = badgeRef.current?.getBoundingClientRect();
    if (!rect) return;
    setPosition({
      left: Math.min(Math.max(rect.left + rect.width / 2, 152), window.innerWidth - 152),
      top: rect.bottom + 8,
    });
    setIsOpen(true);
  };

  const closeTooltip = () => setIsOpen(false);

  return (
    <>
      <span
        ref={badgeRef}
        role="button"
        tabIndex={0}
        aria-label={definition ? `${definition.label} help` : "Parameter help"}
        aria-describedby={isOpen ? tooltipId : undefined}
        className="help-badge"
        onMouseEnter={openTooltip}
        onMouseLeave={closeTooltip}
        onFocus={openTooltip}
        onBlur={closeTooltip}
        onClick={() => (isOpen ? closeTooltip() : openTooltip())}
        onKeyDown={event => {
          if (event.key === "Escape") closeTooltip();
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            isOpen ? closeTooltip() : openTooltip();
          }
        }}
      >
        ?
      </span>
      {isOpen && typeof document !== "undefined" && createPortal(
        <div
          id={tooltipId}
          role="tooltip"
          className="help-tooltip"
          style={{ left: position.left, top: position.top, transform: "translateX(-50%)" }}
        >
          <div className="help-tooltip-title">
            {definition?.label || "Parameter help"}
          </div>
          <div>{definition?.description || title}</div>
          {definition?.unit && (
            <div className="help-tooltip-unit">
              Unit: {definition.unit}
            </div>
          )}
        </div>,
        document.body
      )}
    </>
  );
}
