from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
from typing import Any

def save_bundled_artifact(
    pipeline: Any,
    model: Any,
    explainer: Any,
    reason_mapper: dict[str, str],
    output_dir: Path,
    version: str = "1.0.0"
) -> None:
    """
    Saves the model preprocessing pipeline, classifier, and SHAP explainer
    as a single coupled artifact. This structurally prevents silent drift
    where a deployed model gets evaluated by an outdated explainer.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    bundle = {
        "version": version,
        "pipeline": pipeline,
        "model": model,
        "explainer": explainer,
        "reason_mapper": reason_mapper,
    }
    
    artifact_path = output_dir / "coupled_model_explainer.pkl"
    joblib.dump(bundle, artifact_path)
        
    # Create checksum for robust load checks
    checksum = hashlib.md5(artifact_path.read_bytes()).hexdigest()
    with (output_dir / "artifact_manifest.json").open("w") as fp:
        json.dump({"checksum": checksum, "version": version}, fp)

def load_bundled_artifact(artifact_path: Path, expected_version: str = "1.0.0") -> dict[str, Any]:
    """
    Loads the coupled artifact enforcing explicit version checks to prevent
    dangerous mismatch in production inference.
    """
    bundle = joblib.load(artifact_path)
        
    if bundle.get("version") != expected_version:
        raise ValueError(
            f"CRITICAL ERROR: Artifact version drift detected! "
            f"Expected {expected_version}, got {bundle.get('version')}."
        )
        
    if "pipeline" not in bundle or "explainer" not in bundle:
        raise ValueError("CRITICAL ERROR: Corrupted artifact bundle missing core components.")
        
    return bundle
