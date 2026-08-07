"use client";

// QuEst's tour: which selectors to spotlight and the copy for each step, all
// pulled through `t()` so German and French get the same guide. The heavy
// lifting lives in the reusable ./tour/Tour component; this file is purely the
// QuEst-shaped content.
//
// Steps may route to the page they explain via `before` — the sidebar panels
// change with the active tab — so the tour can walk the whole app without the
// user touching the keyboard. Every `<nav>`/sidebar element is persistent in
// AppLayout (it survives tab changes), which is why most targets reuse the few
// stable `data-tour` hooks and only the page bodies are reached per-page.

import type { TourLabels, TourStep } from "./Tour";

type Translate = (key: string, fallback: string) => string;
type Navigate = (path: string) => void;

export function getTourSteps(t: Translate, navigate: Navigate): TourStep[] {
  return [
    {
      target: "",
      title: t("web.tour.welcome.title", "Welcome to QuEst"),
      body: t(
        "web.tour.welcome.body",
        "This short tour walks you through the whole workflow: loading a protein, placing a dye, running and reading a simulation, scanning residues and tuning the advanced settings. Use the arrow keys or the buttons to move on."
      ),
      before: () => navigate("/labeling"),
    },
    {
      target: '[data-tour="tabs"]',
      title: t("web.tour.step1.title", "Workflow tabs"),
      body: t(
        "web.tour.step1.body",
        "Work through the stages left to right: Labeling, Simulation, Scan and Advanced Settings. The active stage is highlighted and drives which settings the sidebar shows."
      ),
    },
    {
      target: '[data-tour="pdb"]',
      title: t("web.tour.step2.title", "Load a structure"),
      body: t(
        "web.tour.step2.body",
        "Type a four-letter PDB ID and press Load, or upload your own PDB/CIF file. QuEst resolves chains, residues and atoms from it."
      ),
    },
    {
      target: '[data-tour="upload"]',
      title: t("web.tour.step3.title", "Upload a file"),
      body: t(
        "web.tour.step3.body",
        "Prefer a local file over a PDB ID? Upload a PDB or CIF here; the structure replaces the loaded one and the labeling selects repopulate."
      ),
    },
    {
      target: '[data-tour="runs"]',
      title: t("web.tour.step4.title", "Simulation runs"),
      body: t(
        "web.tour.step4.body",
        "Every simulation is saved. Pick a previous run here to reload its labeling site, parameters and results."
      ),
    },
    {
      target: '[data-tour="structure"]',
      title: t("web.tour.step5.title", "Structure panel"),
      body: t(
        "web.tour.step5.body",
        "Confirms what is loaded: the filename plus chain and atom counts. Nothing will run until this shows a structure."
      ),
    },
    {
      target: '[data-tour="labeling"]',
      title: t("web.tour.step6.title", "Place the dye"),
      body: t(
        "web.tour.step6.body",
        "Pick the chain, residue and atom the dye attaches to. A dye cannot be tethered to waters or ions at the end of the structure."
      ),
    },
    {
      target: '[data-tour="fret"]',
      title: t("web.tour.step7.title", "FRET acceptor"),
      body: t(
        "web.tour.step7.body",
        "Optionally add a second dye and define an acceptor site. The Forster-based energy transfer is then combined with the PET quenching in the decay."
      ),
    },
    {
      target: '[data-tour="av"]',
      title: t("web.tour.step8.title", "Accessible volume"),
      body: t(
        "web.tour.step8.body",
        "Defines the space the linker lets the dye explore around the attachment atom. The linker length is the default value the simulation uses."
      ),
    },
    {
      target: '[data-tour="viewer"]',
      title: t("web.tour.step9.title", "Structure viewer"),
      body: t(
        "web.tour.step9.body",
        "This 3D view draws the structure and the labelling site. It updates as you change the chain, residue and atom selects."
      ),
    },
    {
      target: '[data-tour="run"]',
      title: t("web.tour.step10.title", "Run the simulation"),
      body: t(
        "web.tour.step10.body",
        "Once a structure is loaded, hit Run Sim. The decay, quenching and dye trajectory appear on the Simulation tab."
      ),
    },
    {
      target: '[data-tour="sim-settings"]',
      title: t("web.tour.step11.title", "Simulation settings"),
      body: t(
        "web.tour.step11.body",
        "Lifetime, viscosity and the other model parameters live here. The value next to each field is what the current run used."
      ),
      before: () => navigate("/simulation"),
    },
    {
      target: '[data-tour="sim-page"]',
      title: t("web.tour.step12.title", "Simulation results"),
      body: t(
        "web.tour.step12.body",
        "The Simulation tab reads the decays, the quenching-rate histograms and the dye trajectory for the focused site and its acceptors."
      ),
    },
    {
      target: '[data-tour="scan-settings"]',
      title: t("web.tour.step13.title", "Scan settings"),
      body: t(
        "web.tour.step13.body",
        "Choose which residues to evaluate — a single site or every residue in the chain — before running the batch scan."
      ),
      before: () => navigate("/scan"),
    },
    {
      target: '[data-tour="scan-page"]',
      title: t("web.tour.step14.title", "Scan results"),
      body: t(
        "web.tour.step14.body",
        "The scan table reports lifetime, quantum yield, contact fraction and the accessible/contact volumes per residue. Columns sort and filter, and the results export to the clipboard."
      ),
    },
    {
      target: '[data-tour="quench-editor"]',
      title: t("web.tour.step15.title", "Quenching parameters"),
      body: t(
        "web.tour.step15.body",
        "Under Advanced Settings you can edit the per-residue quenching rates that drive the PET model."
      ),
      before: () => navigate("/advanced"),
    },
    {
      target: '[data-tour="adv-config"]',
      title: t("web.tour.step16.title", "Simulation configuration"),
      body: t(
        "web.tour.step16.body",
        "The settings that shape how a simulation is computed — confirms and advanced model switches live here."
      ),
    },
    {
      target: '[data-tour="stickiness"]',
      title: t("web.tour.step17.title", "Dye stickiness"),
      body: t(
        "web.tour.step17.body",
        "Dyes slow down inside the protein, not just at it; the slowing radius captures that transient and is a run parameter you can return to."
      ),
    },
    {
      target: '[data-tour="json"]',
      title: t("web.tour.step18.title", "JSON mode"),
      body: t(
        "web.tour.step18.body",
        "Flip Configuration to JSON Mode to edit the raw project object instead of the curated form. Both drive the same state."
      ),
    },
    {
      target: '[data-tour="preset"]',
      title: t("web.tour.step19.title", "Save a preset"),
      body: t(
        "web.tour.step19.body",
        "Save the current parameters as a named dye preset you can bring back with the Dye selector, instead of re-typing values."
      ),
    },
    {
      target: '[data-tour="language"]',
      title: t("web.tour.step20.title", "Language"),
      body: t(
        "web.tour.step20.body",
        "Switch the whole interface — labels, help text and this tour — between English, German and French. Your choice is remembered."
      ),
    },
    {
      target: '[data-tour="theme"]',
      title: t("web.tour.step21.title", "Theme"),
      body: t(
        "web.tour.step21.body",
        "Toggle between the light and dark appearance. The theme is saved for next visit."
      ),
    },
    {
      target: '[data-tour="cite"]',
      title: t("web.tour.step22.title", "Cite QuEst"),
      body: t(
        "web.tour.step22.body",
        "Click here to see exactly how to cite QuEst in a paper."
      ),
    },
    {
      target: '[data-tour="download"]',
      title: t("web.tour.step23.title", "Download"),
      body: t(
        "web.tour.step23.body",
        "Finally, download the results of the current run as a file, or Clear to reset the application and start again."
      ),
    },
    {
      target: "",
      title: t("web.tour.done.title", "All set"),
      body: t(
        "web.tour.done.body",
        "That is the full workflow: load a structure, place a dye, run a simulation, scan residues and fine-tune the model. You can reopen this tour any time from the button at the top."
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