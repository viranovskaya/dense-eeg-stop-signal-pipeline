"""Provenance-bound ICA review packages for BIDS/EEGLAB recordings."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from .bids_eeglab import (
    BIDSEeglabInputs,
    _captured_json_sha,
    input_identities,
    resolve_inputs,
)
from .bids_eeglab_processing import (
    VerifiedReviewBundle,
    load_verified_review_bundle,
    prepare_filtered_reviewed_raw,
)
from .bids_eeglab_review import _nearest_existing_ancestor, _runtime_identity
from .ica import (
    DEFAULT_ICA_CONFIG,
    ICAConfig,
    _decision_template,
    _ica_config_from_payload,
    component_diagnostics,
    fit_ica,
    ica_decimation,
    save_review_figures,
)
from .provenance import (
    canonical_sha256,
    sha256_file,
    source_manifest,
    verify_provenance,
    write_provenance,
)


def _review_identity(root: Path, review: VerifiedReviewBundle) -> dict:
    provenance = verify_provenance(root)
    if sha256_file(root / "provenance.json") != review.identity["provenance_sha256"]:
        raise ValueError("Decision bundle provenance changed")
    if provenance.get("core_sha256") != review.identity["core_sha256"]:
        raise ValueError("Decision bundle core changed")
    return dict(review.identity)


def _verify_context(
    *,
    inputs: BIDSEeglabInputs,
    profile_path: Path,
    profile_sha256: str,
    qc_path: Path,
    bundle_path: Path,
    participant_id: str,
    review_identity: dict,
    initial_inputs: list[dict],
    initial_source: dict,
    initial_runtime: dict,
    ica_config_path: Path,
    ica_config_sha256: str,
    output: Path | None = None,
) -> None:
    if input_identities(resolve_inputs(inputs)) != initial_inputs:
        raise ValueError("Dataset inputs changed during ICA review")
    if sha256_file(profile_path) != profile_sha256:
        raise ValueError("Processing profile changed during ICA review")
    if sha256_file(ica_config_path) != ica_config_sha256:
        raise ValueError("ICA configuration changed during ICA review")
    if source_manifest() != initial_source or _runtime_identity() != initial_runtime:
        raise ValueError("Source or runtime changed during ICA review")
    current_review = load_verified_review_bundle(
        inputs, qc_path, bundle_path, participant_id
    )
    if current_review.identity != review_identity:
        raise ValueError("Decision bundle changed during ICA review")
    if output is not None:
        verify_provenance(output)


def _run_ica_review(
    inputs: BIDSEeglabInputs,
    profile_path: Path,
    qc_path: Path,
    bundle_path: Path,
    participant_id: str,
    profile_sha256: str,
    config: ICAConfig,
    ica_config_path: Path,
    ica_config_sha256: str,
    output: Path,
) -> dict:
    profile, observed_profile_sha256 = _captured_json_sha(profile_path)
    if observed_profile_sha256 != profile_sha256:
        raise ValueError("Processing profile changed before ICA preparation")
    initial_source = source_manifest()
    initial_runtime = _runtime_identity()
    initial_inputs = input_identities(resolve_inputs(inputs))
    raw, review, processing = prepare_filtered_reviewed_raw(
        inputs,
        profile_path,
        qc_path,
        bundle_path,
        participant_id,
        "ica",
        expected_profile_sha256=profile_sha256,
    )
    review_identity = _review_identity(bundle_path, review)
    if sha256_file(profile_path) != profile_sha256:
        raise ValueError("Processing profile changed during ICA preparation")

    ica = fit_ica(raw, config)
    output.mkdir(parents=True, exist_ok=False)
    solution_path = output / "ica_solution.fif"
    ica.save(solution_path, overwrite=False, verbose="ERROR")
    solution_sha256 = sha256_file(solution_path)
    diagnostics = component_diagnostics(ica, raw, config)
    diagnostics.to_csv(output / "component_diagnostics.csv", index=False)
    figures = save_review_figures(ica, raw, diagnostics, output)
    _decision_template(
        participant_id, solution_sha256, int(ica.n_components_)
    ).to_csv(output / "ica_decision_template.csv", index=False)

    summary = {
        "schema_version": "1",
        "status": "manual_component_review_required",
        "participant_id": participant_id,
        "components": int(ica.n_components_),
        "estimated_eeg_rank": int(ica.n_components_),
        "fit_iterations": int(ica.n_iter_),
        "fit_stopping_rule_met": True,
        "fit_stopping_rule": (
            "MNE small-angle stopping rule "
            f"(n_small_angle={config.fit_params['n_small_angle']})"
        ),
        "decimation": ica_decimation(raw, config),
        "effective_fit_sampling_hz": (
            float(raw.info["sfreq"]) / ica_decimation(raw, config)
        ),
        "review_candidates": int(diagnostics["candidate_review"].sum()),
        "auxiliary_correlation_cues": {
            "eog_channel_available": "EOG" in raw.ch_names,
            "ecg_channel_available": "ECG" in raw.ch_names,
            "missing_channels_produce_nan_not_low_risk": True,
        },
        "variance_denominator": {
            "samples_used_outside_bad_annotations": int(
                diagnostics["variance_samples_used"].iloc[0]
            ),
            "samples_total": int(diagnostics["variance_samples_total"].iloc[0]),
            "bad_samples_omitted": int(
                diagnostics["variance_bad_samples_omitted"].iloc[0]
            ),
        },
        "review_figures": figures,
        "automatic_component_exclusion": False,
        "decision_status": "pending explicit keep or exclude for every component",
        "processing": processing,
        "classification": "controlled_derived",
        "publication_allowed": False,
    }
    (output / "ica_review_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# ICA component review\n\n"
        "> Controlled derived output. Review before sharing or publishing.\n\n"
        "The solution was fitted after provenance-bound channel and segment "
        "review, guarded FIR filtering, and average rereferencing. Correlation "
        "flags are review cues, not component labels or removal decisions. "
        "Properties are provided for every component. Missing EOG or ECG "
        "channels yield unavailable correlation cues, not evidence of low risk. "
        "Copy `ica_decision_template.csv` outside this immutable package and "
        "record an explicit `keep` or `exclude` decision for every component.\n",
        encoding="utf-8",
    )
    core = {
        "schema_version": "1",
        "workflow": "bids_eeglab_ica_review",
        "participant_id": participant_id,
        "inputs": initial_inputs,
        "controls": {
            "profile_sha256": profile_sha256,
            "profile_processing_sha256": canonical_sha256(profile["processing"]),
            "ica_config_sha256": ica_config_sha256,
            "source_manifest": initial_source,
        },
        "runtime": initial_runtime,
        "decision_bundle": review_identity,
        "processing_sha256": canonical_sha256(processing),
        "solution_sha256": solution_sha256,
        "summary_sha256": canonical_sha256(summary),
        "automatic_component_exclusion": False,
        "classification": "controlled_derived",
        "publication_allowed": False,
    }
    core["core_sha256"] = canonical_sha256(core)
    _verify_context(
        inputs=inputs,
        profile_path=profile_path,
        profile_sha256=profile_sha256,
        qc_path=qc_path,
        bundle_path=bundle_path,
        participant_id=participant_id,
        review_identity=review_identity,
        initial_inputs=initial_inputs,
        initial_source=initial_source,
        initial_runtime=initial_runtime,
        ica_config_path=ica_config_path,
        ica_config_sha256=ica_config_sha256,
    )
    write_provenance(output, core)
    verify_provenance(output)
    return summary


def publish_ica_review(
    inputs: BIDSEeglabInputs,
    profile_path: Path,
    qc_path: Path,
    bundle_path: Path,
    participant_id: str,
    output: Path,
    ica_config_path: Path = DEFAULT_ICA_CONFIG,
) -> Path:
    """Publish one exact-set ICA review package atomically to a new directory."""
    profile_path = Path(profile_path).expanduser().absolute()
    qc_path = Path(qc_path).expanduser().absolute()
    bundle_path = Path(bundle_path).expanduser().absolute()
    ica_config_path = Path(ica_config_path).expanduser().absolute()
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("ICA review requires a new output path")
    protected = [
        resolve_inputs(inputs).dataset_root.resolve(strict=True),
        qc_path.resolve(strict=True),
        bundle_path.resolve(strict=True),
    ]
    nearest = _nearest_existing_ancestor(output.parent).resolve(strict=True)
    if any(nearest.is_relative_to(root) for root in protected):
        raise ValueError("ICA review output must remain outside source and review inputs")

    _, profile_sha256 = _captured_json_sha(profile_path)
    config_payload, ica_config_sha256 = _captured_json_sha(ica_config_path)
    config = _ica_config_from_payload(config_payload)
    initial_source = source_manifest()
    initial_runtime = _runtime_identity()
    initial_inputs = input_identities(resolve_inputs(inputs))
    review = load_verified_review_bundle(inputs, qc_path, bundle_path, participant_id)
    review_identity = dict(review.identity)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    temporary.rmdir()
    published = False
    try:
        _run_ica_review(
            inputs,
            profile_path,
            qc_path,
            bundle_path,
            participant_id,
            profile_sha256,
            config,
            ica_config_path,
            ica_config_sha256,
            temporary,
        )
        _verify_context(
            inputs=inputs,
            profile_path=profile_path,
            profile_sha256=profile_sha256,
            qc_path=qc_path,
            bundle_path=bundle_path,
            participant_id=participant_id,
            review_identity=review_identity,
            initial_inputs=initial_inputs,
            initial_source=initial_source,
            initial_runtime=initial_runtime,
            ica_config_path=ica_config_path,
            ica_config_sha256=ica_config_sha256,
            output=temporary,
        )
        temporary.replace(output)
        published = True
        _verify_context(
            inputs=inputs,
            profile_path=profile_path,
            profile_sha256=profile_sha256,
            qc_path=qc_path,
            bundle_path=bundle_path,
            participant_id=participant_id,
            review_identity=review_identity,
            initial_inputs=initial_inputs,
            initial_source=initial_source,
            initial_runtime=initial_runtime,
            ica_config_path=ica_config_path,
            ica_config_sha256=ica_config_sha256,
            output=output,
        )
        return output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        if published:
            shutil.rmtree(output, ignore_errors=True)
        raise
