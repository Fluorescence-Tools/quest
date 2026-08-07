"use client";

// QuEst's tour: which selectors to spotlight and the copy for each step, all
// pulled through `t()` so German and French get the same guide. The heavy
// lifting lives in the reusable ./tour/Tour component; this file is purely the
// QuEst-shaped content.

import type { TourLabels, TourStep } from "./Tour";

type Translate = (key: string, fallback: string) => string;

export function getTourSteps(t: Translate): TourStep[] {
  return [
    {
      target: '[data-tour="tabs"]',
      title: t("web.tour.step1.title", "Workflow tabs"),
      body: t(
        "web.tour.step1.body",
        "Work through the stages left to right: Labeling, Simulation, Scan and Advanced Settings. The active stage is highlighted for you."
      ),
    },
    {
      target: '[data-tour="structure"]',
      title: t("web.tour.step2.title", "Load a structure"),
      body: t(
        "web.tour.step2.body",
        "Type a four-letter PDB ID and press Load, or upload your own PDB/CIF file. QuEst resolves chains, residues and atoms from it."
      ),
    },
    {
      target: '[data-tour="labeling"]',
      title: t("web.tour.step3.title", "Place the dye"),
      body: t(
        "web.tour.step3.body",
        "Pick the chain, residue and atom the dye attaches to. A dye cannot be tethered to waters or ions at the end of the structure."
      ),
    },
    {
      target: '[data-tour="run"]',
      title: t("web.tour.step4.title", "Run the simulation"),
      body: t(
        "web.tour.step4.body",
        "Once a structure is loaded, run the simulation and read the decay, the quenching and the dye trajectory in the results."
      ),
    },
  ];
}

export function getTourLabels(t: Translate): TourLabels {
  const stepOf = t("web.tour.step_of", "Step {c} of {t}");
  return {
    next: t("web.tour.next", "Next"),
    prev: t("web.tour.prev", "Back"),
    done: t("web.tour.done", "Done"),
    close: t("web.tour.close", "Close tour"),
    stepIndicator: (c: number, total: number) =>
      stepOf.replace("{c}", String(c)).replace("{t}", String(total)),
  };
}