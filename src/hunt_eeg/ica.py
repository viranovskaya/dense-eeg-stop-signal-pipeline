"""Explicit ICA review and application for dense EEG recordings."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from .config import DEFAULT_ANALYSIS_CONFIG, AnalysisConfig, load_analysis_config
from .provenance import (
    build_provenance_core,
    canonical_sha256,
    sha256_file,
    source_manifest,
    verify_input_sources,
    verify_provenance,
    write_provenance,
)
from .qc import _read_brainvision_compat

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ICA_CONFIG = PROJECT_ROOT / "config" / "ica.json"

ICA_DECISION_COLUMNS = [
    "participant_id",
    "ica_solution_sha256",
    "component",
    "decision",
    "reason",
    "reviewer",
    "reviewed_at",
    "evidence",
]


@dataclass(frozen=True)
class ICAConfig:
    """Parameters that make ICA fitting repeatable."""

    method: str
    fit_params: dict[str, bool | int | float]
    random_state: int
    max_iter: int
    fit_sampling_hz: float
    correlation_review_threshold: float


def load_ica_config(path: Path = DEFAULT_ICA_CONFIG) -> ICAConfig:
    """Load the bounded ICA configuration used for review packages."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "method",
        "fit_params",
        "random_state",
        "max_iter",
        "fit_sampling_hz",
        "correlation_review_threshold",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"ICA config is missing required fields: {missing}")
    if payload["method"] != "infomax":
        raise ValueError("Only the reviewed extended Infomax workflow is supported")
    fit_params = payload["fit_params"]
    expected_fit_params = {
        "extended": True,
        "n_small_angle": 20,
        "w_change": 0.0,
    }
    if fit_params != expected_fit_params:
        raise ValueError(
            "ICA fit_params must request extended Infomax with the declared "
            "small-angle stopping rule"
        )
    random_state = int(payload["random_state"])
    max_iter = int(payload["max_iter"])
    fit_sampling_hz = float(payload["fit_sampling_hz"])
    threshold = float(payload["correlation_review_threshold"])
    if max_iter <= 0 or not math.isfinite(fit_sampling_hz) or fit_sampling_hz <= 0:
        raise ValueError("ICA max_iter and fit sampling frequency must be positive")
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("ICA correlation threshold must be between zero and one")
    return ICAConfig(
        method="infomax",
        fit_params=dict(expected_fit_params),
        random_state=random_state,
        max_iter=max_iter,
        fit_sampling_hz=fit_sampling_hz,
        correlation_review_threshold=threshold,
    )


def _participant_id(value: str) -> str:
    value = str(value)
    if not value or len(value) > 64 or not value.isascii() or not value.isalnum():
        raise ValueError("Participant ID must be 1-64 ASCII letters or digits")
    return value


def _reviewed_bad_channels(decisions: list[dict], raw: mne.io.BaseRaw) -> list[str]:
    decision_channels = {
        str(decision.get("channel", ""))
        for decision in decisions
        if decision.get("channel")
    }
    unknown = sorted(decision_channels - set(raw.ch_names))
    if unknown:
        raise ValueError(f"Bad-channel labels not present in recording: {unknown}")
    bads = sorted(
        str(decision["channel"])
        for decision in decisions
        if decision.get("decision") == "interpolate"
    )
    non_eeg = [
        channel
        for channel in bads
        if raw.get_channel_types(picks=[channel])[0] != "eeg"
    ]
    if non_eeg:
        raise ValueError(f"Only scalp EEG channels may be marked bad: {non_eeg}")
    return bads


def prepare_ica_input(
    vhdr: Path,
    bad_channel_decisions: list[dict],
    analysis: AnalysisConfig | None = None,
) -> mne.io.BaseRaw:
    """Load, type, filter and reference one recording for ICA fitting."""
    analysis = load_analysis_config() if analysis is None else analysis
    raw = _read_brainvision_compat(Path(vhdr).expanduser().resolve())
    type_updates = {
        name: kind
        for name, kind in analysis.channel_types.items()
        if name in raw.ch_names
    }
    raw.set_channel_types(type_updates, verbose="ERROR")
    raw.set_montage(
        mne.channels.make_standard_montage(analysis.montage),
        match_case=False,
        on_missing="warn",
        verbose="ERROR",
    )
    bads = _reviewed_bad_channels(bad_channel_decisions, raw)
    raw.load_data(verbose="ERROR")
    low_hz, high_hz = analysis.filter_hz
    if high_hz >= float(raw.info["sfreq"]) / 2:
        raise ValueError("ICA input filter must remain below the recording Nyquist")
    picks = mne.pick_types(raw.info, eeg=True, eog=True, ecg=False, exclude=[])
    raw.filter(
        low_hz,
        high_hz,
        picks=picks,
        method="fir",
        phase="zero",
        verbose="ERROR",
    )
    raw.info["bads"] = bads
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    return raw


def ica_decimation(raw: mne.io.BaseRaw, config: ICAConfig) -> int:
    """Return a safe integer decimation for the filtered ICA input."""
    sfreq = float(raw.info["sfreq"])
    lowpass = float(raw.info["lowpass"])
    desired = max(1, round(sfreq / config.fit_sampling_hz))
    maximum_safe = max(1, math.floor(sfreq / (2.5 * lowpass)))
    decim = min(desired, maximum_safe)
    effective_sfreq = sfreq / decim
    if effective_sfreq < 2.5 * lowpass:
        raise ValueError(
            "Effective ICA fit sampling frequency must be at least 2.5 times "
            "the input low-pass"
        )
    return decim


def fit_ica(raw: mne.io.BaseRaw, config: ICAConfig) -> mne.preprocessing.ICA:
    """Fit rank-aware extended Infomax to reviewed good EEG channels."""
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if len(eeg_picks) < 2:
        raise ValueError("ICA requires at least two reviewed good EEG channels")
    rank_input = raw.copy().pick(eeg_picks)
    rank = mne.compute_rank(
        rank_input,
        rank=None,
        tol="auto",
        verbose="ERROR",
    ).get("eeg", 0)
    if not 1 < rank <= len(eeg_picks):
        raise ValueError(f"Invalid estimated EEG rank for ICA: {rank}")
    ica = mne.preprocessing.ICA(
        n_components=int(rank),
        method=config.method,
        fit_params=dict(config.fit_params),
        random_state=config.random_state,
        max_iter=config.max_iter,
    )
    ica.fit(
        raw,
        picks=eeg_picks,
        decim=ica_decimation(raw, config),
        reject_by_annotation=True,
        verbose="ERROR",
    )
    if int(ica.n_iter_) >= config.max_iter:
        raise RuntimeError(
            "Extended Infomax did not meet the declared small-angle stopping "
            f"rule within {config.max_iter} iterations"
        )
    return ica


def _target_scores(
    ica: mne.preprocessing.ICA,
    raw: mne.io.BaseRaw,
    target: str,
) -> np.ndarray:
    if target not in raw.ch_names:
        return np.full(ica.n_components_, np.nan)
    scores = ica.score_sources(
        raw,
        target=target,
        score_func="pearsonr",
        verbose="ERROR",
    )
    return np.asarray(scores, dtype=float)


def component_diagnostics(
    ica: mne.preprocessing.ICA,
    raw: mne.io.BaseRaw,
    config: ICAConfig,
) -> pd.DataFrame:
    """Return review cues without converting them into removal decisions."""
    eog_scores = _target_scores(ica, raw, "EOG")
    ecg_scores = _target_scores(ica, raw, "ECG")
    sources = ica.get_sources(raw).get_data()
    source_variance = np.var(sources, axis=1)
    variance_total = float(source_variance.sum())
    variance_fraction = (
        source_variance / variance_total
        if variance_total > 0
        else np.full(len(source_variance), np.nan)
    )
    threshold = config.correlation_review_threshold
    rows = []
    for component in range(ica.n_components_):
        eog = float(eog_scores[component])
        ecg = float(ecg_scores[component])
        candidate_eog = bool(np.isfinite(eog) and abs(eog) >= threshold)
        candidate_ecg = bool(np.isfinite(ecg) and abs(ecg) >= threshold)
        rows.append(
            {
                "component": component,
                "eog_correlation": eog,
                "ecg_correlation": ecg,
                "source_variance_fraction": float(variance_fraction[component]),
                "candidate_eog": candidate_eog,
                "candidate_ecg": candidate_ecg,
                "candidate_review": candidate_eog or candidate_ecg,
                "automatic_exclusion": False,
            }
        )
    return pd.DataFrame(rows)


def _decision_template(
    participant_id: str,
    solution_sha256: str,
    components: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "participant_id": participant_id,
                "ica_solution_sha256": solution_sha256,
                "component": component,
                "decision": "",
                "reason": "",
                "reviewer": "",
                "reviewed_at": "",
                "evidence": "",
            }
            for component in range(components)
        ],
        columns=ICA_DECISION_COLUMNS,
    )


def _as_figures(value) -> list:
    return list(value) if isinstance(value, list) else [value]


def save_review_figures(
    ica: mne.preprocessing.ICA,
    raw: mne.io.BaseRaw,
    diagnostics: pd.DataFrame,
    output: Path,
) -> list[str]:
    """Save an overview and detailed plots for correlation-based candidates."""
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    page_size = 20
    for page, start in enumerate(
        range(0, ica.n_components_, page_size),
        start=1,
    ):
        picks = list(range(start, min(start + page_size, ica.n_components_)))
        overviews = _as_figures(
            ica.plot_components(
                picks=picks,
                inst=raw,
                nrows=4,
                ncols=5,
                res=48,
                show=False,
                verbose="ERROR",
            )
        )
        for figure_index, figure in enumerate(overviews, start=1):
            suffix = "" if len(overviews) == 1 else f"-{figure_index:02d}"
            path = figures / f"component_topographies_page-{page:02d}{suffix}.png"
            figure.savefig(path, dpi=140, bbox_inches="tight")
            plt.close(figure)
            paths.append(path.relative_to(output).as_posix())
    candidates = (
        diagnostics.loc[diagnostics["candidate_review"], "component"]
        .astype(int)
        .tolist()
    )
    if candidates:
        properties = ica.plot_properties(
            raw,
            picks=candidates,
            psd_args={"fmax": 40.0},
            show=False,
            verbose="ERROR",
        )
        for component, figure in zip(candidates, properties, strict=True):
            path = figures / f"component-{component:03d}_properties.png"
            figure.savefig(path, dpi=170, bbox_inches="tight")
            plt.close(figure)
            paths.append(path.relative_to(output).as_posix())
    return paths


def _run_ica_review_into(
    vhdr: Path,
    output: Path,
    participant_id: str,
    bad_channel_decisions: list[dict],
) -> dict:
    vhdr = Path(vhdr).expanduser().resolve()
    participant_id = _participant_id(participant_id)
    output.mkdir(parents=True, exist_ok=False)
    initial_source = source_manifest()
    initial_analysis_sha256 = sha256_file(DEFAULT_ANALYSIS_CONFIG)
    initial_ica_sha256 = sha256_file(DEFAULT_ICA_CONFIG)
    analysis = load_analysis_config()
    config = load_ica_config()
    if (
        sha256_file(DEFAULT_ANALYSIS_CONFIG) != initial_analysis_sha256
        or sha256_file(DEFAULT_ICA_CONFIG) != initial_ica_sha256
        or source_manifest()["sha256"] != initial_source["sha256"]
    ):
        raise RuntimeError("ICA source or configuration changed during initialization")
    core = build_provenance_core(
        vhdr=vhdr,
        participant_id=participant_id,
        analysis=analysis,
        bad_channel_decisions=bad_channel_decisions,
    )
    if core["software"]["source_manifest"]["sha256"] != initial_source["sha256"]:
        raise RuntimeError("ICA provenance source does not match the fitted source")
    if core["configuration"]["analysis_sha256"] != initial_analysis_sha256:
        raise RuntimeError("ICA provenance analysis config does not match the fit")
    raw = prepare_ica_input(vhdr, bad_channel_decisions, analysis)
    core_without_hash = {
        key: value for key, value in core.items() if key != "core_sha256"
    }
    core_without_hash["workflow"] = "ica_component_review"
    core_without_hash["ica_configuration"] = {
        "config_sha256": initial_ica_sha256,
        "method": config.method,
        "fit_params": config.fit_params,
        "random_state": config.random_state,
        "max_iter": config.max_iter,
        "fit_sampling_hz": config.fit_sampling_hz,
        "decim": ica_decimation(raw, config),
        "effective_fit_sampling_hz": (
            float(raw.info["sfreq"]) / ica_decimation(raw, config)
        ),
        "correlation_review_threshold": config.correlation_review_threshold,
        "automatic_exclusion": False,
    }
    core = {**core_without_hash, "core_sha256": canonical_sha256(core_without_hash)}

    ica = fit_ica(raw, config)
    solution = output / "ica_solution.fif"
    ica.save(solution, overwrite=False, verbose="ERROR")
    solution_sha256 = sha256_file(solution)
    diagnostics = component_diagnostics(ica, raw, config)
    diagnostics.to_csv(output / "component_diagnostics.csv", index=False)
    figure_paths = save_review_figures(ica, raw, diagnostics, output)
    _decision_template(
        participant_id,
        solution_sha256,
        ica.n_components_,
    ).to_csv(output / "ica_decision_template.csv", index=False)
    summary = {
        "participant_id": participant_id,
        "source_header": "withheld-private-source",
        "components": int(ica.n_components_),
        "estimated_eeg_rank": int(ica.n_components_),
        "fit_iterations": int(ica.n_iter_),
        "fit_stopping_rule_met": True,
        "fit_stopping_rule": (
            "MNE small-angle stopping rule "
            f"(n_small_angle={config.fit_params['n_small_angle']})"
        ),
        "review_candidates": int(diagnostics["candidate_review"].sum()),
        "review_figures": figure_paths,
        "automatic_exclusion": False,
        "decision_status": "pending explicit keep/exclude decision for every component",
        "privacy": "Private derived output; review before sharing or publishing.",
    }
    (output / "ica_review_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# ICA component review\n\n"
        "This is a private derived review package. Correlations are review cues, "
        "not component labels and not removal decisions. Inspect the component "
        "topography, time course, spectrum, EOG/ECG relationship and effect on "
        "the EEG. Copy `ica_decision_template.csv` outside this immutable package "
        "before editing it. Every component must receive an explicit `keep` or "
        "`exclude` decision.\n",
        encoding="utf-8",
    )
    verify_input_sources(vhdr, core["inputs"])
    if (
        sha256_file(DEFAULT_ANALYSIS_CONFIG) != initial_analysis_sha256
        or sha256_file(DEFAULT_ICA_CONFIG) != initial_ica_sha256
        or source_manifest()["sha256"] != initial_source["sha256"]
    ):
        raise RuntimeError("ICA source or configuration changed during the run")
    write_provenance(output, core)
    verify_provenance(output)
    return summary


def run_ica_review(
    vhdr: Path,
    output: Path,
    participant_id: str,
    bad_channel_decisions: list[dict],
) -> dict:
    """Create one private ICA review package atomically in a new directory."""
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("ICA review requires a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    temporary.rmdir()
    try:
        summary = _run_ica_review_into(
            vhdr=vhdr,
            output=temporary,
            participant_id=participant_id,
            bad_channel_decisions=bad_channel_decisions,
        )
        verify_provenance(temporary)
        temporary.replace(output)
        return summary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_ica_decisions(
    path: Path,
    *,
    participant_id: str,
    solution_sha256: str,
    components: int,
) -> pd.DataFrame:
    """Require a complete, solution-bound decision for every ICA component."""
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = sorted(set(ICA_DECISION_COLUMNS) - set(table.columns))
    if missing:
        raise ValueError("ICA decision table is missing columns: " + ", ".join(missing))
    table = table[ICA_DECISION_COLUMNS].copy()
    for column in ICA_DECISION_COLUMNS:
        table[column] = table[column].str.strip()
    selected = table.loc[
        table["participant_id"] == _participant_id(participant_id)
    ].copy()
    if len(selected) != components:
        raise ValueError(
            f"ICA decisions must contain exactly {components} rows for participant"
        )
    if set(selected["ica_solution_sha256"]) != {solution_sha256}:
        raise ValueError("ICA decision table does not match the solution SHA-256")
    try:
        selected["component"] = selected["component"].astype(int)
    except ValueError as error:
        raise ValueError("ICA component labels must be integers") from error
    if selected["component"].duplicated().any() or set(selected["component"]) != set(
        range(components)
    ):
        raise ValueError("ICA decisions must cover every component exactly once")
    invalid = sorted(set(selected["decision"]) - {"keep", "exclude"})
    if invalid:
        raise ValueError(f"Invalid ICA decisions: {invalid}")
    required_text = ["reason", "reviewer", "reviewed_at", "evidence"]
    empty = [column for column in required_text if (selected[column] == "").any()]
    if empty:
        raise ValueError(
            "ICA decision table contains empty fields: " + ", ".join(empty)
        )
    for value in selected["reviewed_at"].unique():
        try:
            date.fromisoformat(value)
        except ValueError as error:
            raise ValueError(
                f"ICA review date must use YYYY-MM-DD: {value!r}"
            ) from error
    return selected.sort_values("component").reset_index(drop=True)


def apply_reviewed_ica(
    raw: mne.io.BaseRaw,
    *,
    participant_id: str,
    solution_path: Path,
    decision_path: Path,
    expected_context: dict,
) -> tuple[mne.io.BaseRaw, dict]:
    """Apply only components explicitly excluded in the bound review table."""
    solution_path = Path(solution_path).expanduser().resolve()
    decision_path = Path(decision_path).expanduser().resolve()
    review_payload = verify_provenance(solution_path.parent)
    if review_payload.get("workflow") != "ica_component_review":
        raise ValueError("ICA solution is not part of a verified review package")
    if review_payload.get("participant_id") != _participant_id(participant_id):
        raise ValueError("ICA review package belongs to another participant")
    context_checks = {
        "inputs": review_payload.get("inputs"),
        "analysis_sha256": review_payload.get("configuration", {}).get(
            "analysis_sha256"
        ),
        "bad_channel_decisions": review_payload.get("bad_channel_decisions"),
        "source_manifest_sha256": review_payload.get("software", {})
        .get("source_manifest", {})
        .get("sha256"),
    }
    if context_checks != expected_context:
        raise ValueError(
            "ICA review package does not match the current input, configuration, "
            "channel decisions or executable source"
        )
    solution_sha256 = sha256_file(solution_path)
    ica = mne.preprocessing.read_ica(solution_path, verbose="ERROR")
    missing_channels = sorted(set(ica.ch_names) - set(raw.ch_names))
    if missing_channels:
        raise ValueError(
            f"ICA solution channels are missing from input: {missing_channels}"
        )
    decisions = load_ica_decisions(
        decision_path,
        participant_id=participant_id,
        solution_sha256=solution_sha256,
        components=ica.n_components_,
    )
    excluded = (
        decisions.loc[decisions["decision"] == "exclude", "component"]
        .astype(int)
        .tolist()
    )
    cleaned = raw.copy()
    ica.apply(cleaned, exclude=excluded, verbose="ERROR")
    return cleaned, {
        "status": "applied reviewed decisions",
        "components": int(ica.n_components_),
        "excluded_components": excluded,
        "decision_record_complete": True,
        "automatic_exclusion": False,
        "solution_sha256": solution_sha256,
        "review_provenance_sha256": sha256_file(
            solution_path.parent / "provenance.json"
        ),
        "decision_table_sha256": sha256_file(decision_path),
    }
