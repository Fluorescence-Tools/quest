"use client";

// A drop-in, dependency-free onboarding tour. It is deliberately not a quest
// component: it knows nothing about QuEst, project state or i18n. Give it an
// array of steps (a `data-tour` selector, a title and a body) plus control
// labels that are already translated, and it draws a spotlight over each
// target with a tooltip card. Both FPSIMP and QuEst consume the same component.

import React from "react";

export interface TourStep {
  /** CSS selector for the element to spotlight. `data-tour="…"` attributes on
   *  the host app's elements are the stable and discoverable way to do this. */
  target: string;
  title: string;
  body: string;
}

export interface TourLabels {
  next: string;
  prev: string;
  done: string;
  close: string;
  /** `(current, total)` → "Step 2 of 5". Callers supply the words. */
  stepIndicator: (current: number, total: number) => string;
}

interface TourProps {
  open: boolean;
  steps: TourStep[];
  labels: TourLabels;
  onClose: () => void;
}

interface Rect {
  top: number;
  left: number;
  width: number;
  height: number;
}

const GAP = 12;

function emptyRect(): Rect {
  return { top: 0, left: 0, width: 0, height: 0 };
}

/** Position the tooltip card beside the target's rect, clamped into view. */
function placeCard(rect: Rect | null): { below: boolean; top: number; left: number; width: number } {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const width = Math.min(360, vw - 24);
  let below = true;
  let top = Math.max(12, vh / 3);
  let left = (vw - width) / 2;

  if (rect && rect.width > 0 && rect.height > 0) {
    const under = rect.top + rect.height + GAP;
    const above = rect.top - GAP;
    below = under + 240 <= vh ? true : above - 240 >= 0 ? false : true;
    left = rect.left + rect.width / 2 - width / 2;
    left = Math.max(12, Math.min(left, vw - width - 12));
    top = below
      ? Math.min(under, vh - 240 - 12)
      : Math.max(12, above - 240);
  }
  return { below, top, left, width };
}

export function Tour({ open, steps, labels, onClose }: TourProps) {
  const [index, setIndex] = React.useState(0);
  const [rect, setRect] = React.useState<Rect>(emptyRect);
  const [placement, setPlacement] = React.useState<{
    below: boolean;
    top: number;
    left: number;
    width: number;
  }>(() => ({ below: true, top: 0, left: 0, width: 0 }));

  const step = steps[index] ?? null;
  const total = steps.length;
  const isLast = index === total - 1;

  React.useEffect(() => {
    if (open) setIndex(0);
  }, [open]);

  React.useEffect(() => {
    if (!open || !step || !step.target) return;
    const measure = () => {
      const el = document.querySelector<HTMLElement>(step.target);
      if (!el) {
        setRect(emptyRect());
        return;
      }
      el.scrollIntoView({ block: "center", behavior: "smooth" });
      const r = el.getBoundingClientRect();
      setRect({ top: r.top, left: r.left, width: r.width, height: r.height });
    };
    const t = window.setTimeout(measure, 200);
    window.addEventListener("resize", measure);
    return () => {
      window.clearTimeout(t);
      window.removeEventListener("resize", measure);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, index, steps]);

  React.useEffect(() => {
    setPlacement(placeCard(rect));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rect]);

  React.useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowRight" || e.key === "Enter") advance();
      else if (e.key === "ArrowLeft" && index > 0) setIndex(index - 1);
    };
    function advance() {
      setIndex((i) => {
        if (total === 0) return i;
        if (i === total - 1) {
          onClose();
          return i;
        }
        return i + 1;
      });
    }
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = prev;
      window.removeEventListener("keydown", onKey);
    };
  }, [open, index, total, onClose]);

  if (!open || !step) return null;
  const hasTarget = rect.width > 2 && rect.height > 2;

  return (
    <div
      className="tour-overlay"
      onClick={() => {
        if (isLast) onClose();
        else setIndex(index + 1);
      }}
    >
      {/* The box-shadow spread is the simplest reliable way to cut a spotlight
          out of a dimming layer in one element. */}
      {hasTarget && (
        <div
          className="tour-highlight"
          style={{ top: rect.top, left: rect.left, width: rect.width, height: rect.height }}
        />
      )}
      <div
        className={`tour-card ${placement.below ? "below" : "above"}`}
        role="dialog"
        aria-modal="true"
        aria-label={labels.stepIndicator(index + 1, total)}
        style={{ top: placement.top, left: placement.left, maxWidth: placement.width }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="tour-card-header">
          <span className="tour-step">{labels.stepIndicator(index + 1, total)}</span>
          <button className="tour-close" aria-label={labels.close} onClick={onClose}>
            ×
          </button>
        </div>
        <h3 className="tour-card-title">{step.title}</h3>
        <p className="tour-card-body">{step.body}</p>
        <div className="tour-card-footer">
          <button
            className="tour-btn"
            disabled={index === 0}
            onClick={() => (index > 0 ? setIndex(index - 1) : undefined)}
          >
            {labels.prev}
          </button>
          <button
            className="tour-btn primary"
            onClick={() => (isLast ? onClose() : setIndex(index + 1))}
          >
            {isLast ? labels.done : labels.next}
          </button>
        </div>
      </div>
    </div>
  );
}