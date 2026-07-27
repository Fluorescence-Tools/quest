"use client";

import React from "react";
import { useAppContext } from "../../context/AppContext";
import ProteinViewer from "../../components/ProteinViewer";

export default function LabelingPage() {
  const { structure, project, result, surfaceRendering, renderSparsity, onSequenceResidueSelect } = useAppContext();

  return (
    <div className="viewer-card">
      <div className="card-header">
        <h2>Protein Viewer</h2>
      </div>
      <ProteinViewer
        structure={structure}
        project={project}
        result={result}
        active={true}
        surfaceRendering={surfaceRendering}
        renderSparsity={renderSparsity}
        onResidueSelect={onSequenceResidueSelect}
      />
    </div>
  );
}
