"""Fail-closed inventory for versioned BIDS EEG recordings stored as EEGLAB."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from .provenance import (
    canonical_sha256,
    package_versions,
    sha256_file,
    source_manifest,
    verify_provenance,
    write_provenance,
)


@dataclass(frozen=True)
class BIDSEeglabInputs:
    """Dataset root and one recording selected for BIDS-aware discovery."""

    dataset_root: Path
    raw_set: Path


@dataclass(frozen=True)
class ResolvedBIDSEeglabInputs:
    """Complete effective input set derived from one BIDS recording."""

    dataset_root: Path
    raw_set: Path
    raw_payloads: tuple[Path, ...]
    events_tsv: Path
    channels_tsv: Path
    eeg_json: tuple[Path, ...]
    events_json: tuple[Path, ...]
    dataset_description: Path


def _captured_json_sha(path: Path) -> tuple[dict, str]:
    data = Path(path).read_bytes()
    return json.loads(data.decode("utf-8-sig")), hashlib.sha256(data).hexdigest()


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _source_identity(path: Path, dataset_root: Path, role: str) -> dict:
    """Hash a regular source or a bounded final git-annex symlink."""
    root = Path(dataset_root).expanduser().absolute()
    supplied = Path(path).expanduser().absolute()
    lexical = supplied.parent.resolve(strict=True) / supplied.name
    if not _within(lexical, root):
        raise ValueError(f"{role} is outside the dataset root")
    for parent in lexical.parents:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError(f"{role} has a symlinked parent")
    is_link = lexical.is_symlink()
    resolved = lexical.resolve(strict=True)
    resolved_root = root.resolve(strict=True)
    if not _within(resolved, resolved_root) or not resolved.is_file():
        raise ValueError(f"{role} does not resolve to a regular in-dataset file")
    identity = {
        "role": role,
        "format": lexical.suffix.lower(),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "final_symlink": is_link,
    }
    if is_link:
        annex_root = (resolved_root / ".git" / "annex" / "objects").resolve()
        if not _within(resolved, annex_root):
            raise ValueError(f"{role} is not a bounded git-annex symlink")
        target = lexical.readlink().as_posix().encode("utf-8")
        identity["link_text_sha256"] = hashlib.sha256(target).hexdigest()
    return identity


def _parse_entities(path: Path, suffix: str, extension: str) -> dict[str, str] | None:
    name = path.name
    ending = f"_{suffix}.{extension}"
    if name == f"{suffix}.{extension}":
        prefix = ""
    elif name.endswith(ending):
        prefix = name[: -len(ending)]
    else:
        return None
    entities: dict[str, str] = {}
    for item in filter(None, prefix.split("_")):
        if "-" not in item:
            raise ValueError(f"Invalid BIDS entity in {name}")
        key, value = item.split("-", 1)
        if not key or not value or key in entities:
            raise ValueError(f"Invalid or duplicate BIDS entity in {name}")
        entities[key] = value
    return entities


def _inheritance_chain(
    dataset_root: Path,
    raw_parent: Path,
    raw_entities: dict[str, str],
    suffix: str,
) -> tuple[Path, ...]:
    candidates: list[tuple[int, int, Path]] = []
    current = dataset_root
    directories = [current]
    relative_parent = raw_parent.relative_to(dataset_root)
    for part in relative_parent.parts:
        current = current / part
        directories.append(current)
    for depth, directory in enumerate(directories):
        for path in sorted(directory.glob(f"*{suffix}.json")):
            entities = _parse_entities(path, suffix, "json")
            if entities is None or any(
                raw_entities.get(key) != value for key, value in entities.items()
            ):
                continue
            candidates.append((depth, len(entities), path))
    if not candidates:
        raise FileNotFoundError(f"No applicable {suffix}.json sidecar found")
    keys = [(depth, count) for depth, count, _ in candidates]
    duplicates = {key for key in keys if keys.count(key) > 1}
    if duplicates:
        raise ValueError(f"Ambiguous BIDS inheritance for {suffix}.json")
    return tuple(path for _, _, path in sorted(candidates))


def resolve_inputs(inputs: BIDSEeglabInputs) -> ResolvedBIDSEeglabInputs:
    """Derive exact tabular files, sidecar inheritance, and EEGLAB payloads."""
    root = Path(inputs.dataset_root).expanduser().absolute().resolve(strict=True)
    supplied_raw = Path(inputs.raw_set).expanduser().absolute()
    raw_set = supplied_raw.parent.resolve(strict=True) / supplied_raw.name
    raw_entities = _parse_entities(raw_set, "eeg", "set")
    if raw_entities is None or "sub" not in raw_entities or "task" not in raw_entities:
        raise ValueError("Raw file must be a BIDS *_eeg.set recording")
    _source_identity(raw_set, root, "raw_eeglab_set")
    raw_probe = mne.io.read_raw_eeglab(raw_set, preload=False, verbose="ERROR")
    payload_candidates = {
        Path(filename).expanduser().absolute()
        for filename in raw_probe.filenames
        if filename is not None
    }
    payloads = tuple(
        sorted(
            path
            for path in payload_candidates
            if path.resolve(strict=True) != raw_set.resolve(strict=True)
        )
    )
    prefix = raw_set.name[: -len("_eeg.set")]
    events_tsv = raw_set.with_name(f"{prefix}_events.tsv")
    channels_tsv = raw_set.with_name(f"{prefix}_channels.tsv")
    return ResolvedBIDSEeglabInputs(
        dataset_root=root,
        raw_set=raw_set,
        raw_payloads=payloads,
        events_tsv=events_tsv,
        channels_tsv=channels_tsv,
        eeg_json=_inheritance_chain(root, raw_set.parent, raw_entities, "eeg"),
        events_json=_inheritance_chain(root, raw_set.parent, raw_entities, "events"),
        dataset_description=root / "dataset_description.json",
    )


def _recording_identity(raw_set: Path, profile: dict) -> dict:
    entities = _parse_entities(raw_set, "eeg", "set")
    if entities is None:
        raise ValueError("Cannot derive recording identity from the raw filename")
    expected = profile.get("recording")
    if not isinstance(expected, dict):
        raise TypeError("Profile must bind the selected recording identity")
    participant = str(expected.get("participant_id", ""))
    run = str(expected.get("run", ""))
    if not participant or entities.get("sub") != participant:
        raise ValueError("Raw participant does not match the profile recording")
    if not run or entities.get("run") != run:
        raise ValueError("Raw run does not match the profile recording")
    if entities.get("task") != str(profile["task"]):
        raise ValueError("Raw task does not match the profile")
    return {
        "participant_id": participant,
        "task": str(profile["task"]),
        "run": run,
    }


def _input_paths(inputs: ResolvedBIDSEeglabInputs) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = [
        ("raw_eeglab_set", inputs.raw_set),
        ("events_tsv", inputs.events_tsv),
        ("channels_tsv", inputs.channels_tsv),
        ("dataset_description", inputs.dataset_description),
    ]
    paths.extend(
        (f"raw_eeglab_payload_{index:02d}", path)
        for index, path in enumerate(inputs.raw_payloads, start=1)
    )
    paths.extend(
        (f"eeg_json_{index:02d}", path)
        for index, path in enumerate(inputs.eeg_json, start=1)
    )
    paths.extend(
        (f"events_json_{index:02d}", path)
        for index, path in enumerate(inputs.events_json, start=1)
    )
    return paths


def input_identities(inputs: ResolvedBIDSEeglabInputs) -> list[dict]:
    """Return path-free identities for every consumed dataset file."""
    roles = [role for role, _ in _input_paths(inputs)]
    if len(roles) != len(set(roles)):
        raise ValueError("Input roles must be unique")
    lexical = [Path(path).expanduser().absolute() for _, path in _input_paths(inputs)]
    resolved = [path.resolve(strict=True) for path in lexical]
    if len(lexical) != len(set(lexical)) or len(resolved) != len(set(resolved)):
        raise ValueError("Input files must be unique")
    return [
        _source_identity(path, inputs.dataset_root, role)
        for role, path in _input_paths(inputs)
    ]


def _capture(path: Path, expected_sha256: str) -> bytes:
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Dataset input changed before it was consumed")
    return data


def _merge_json(
    paths: tuple[Path, ...], identities: dict[str, dict], prefix: str
) -> dict:
    merged: dict = {}
    for index, path in enumerate(paths, start=1):
        role = f"{prefix}_{index:02d}"
        payload = json.loads(
            _capture(path, identities[role]["sha256"]).decode("utf-8-sig")
        )
        merged.update(payload)
    return merged


def _read_raw(path: Path) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    if not raw.preload:
        raise RuntimeError("EEGLAB input was not fully loaded")
    return raw


def _event_alignment(
    raw: mne.io.BaseRaw, table: pd.DataFrame, tolerance_samples: float
) -> dict:
    required = {"onset", "sample", "value"}
    if not required.issubset(table.columns):
        raise ValueError("events.tsv must contain onset, sample, and value")
    descriptions = table["value"].astype(str).tolist()
    annotation_descriptions = raw.annotations.description.astype(str).tolist()
    if descriptions != annotation_descriptions:
        raise ValueError("BIDS event values do not match raw annotations in order")
    try:
        onsets = pd.to_numeric(table["onset"], errors="raise").to_numpy(float)
        samples = pd.to_numeric(table["sample"], errors="raise").to_numpy(float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Event onsets and integer samples must be finite and nonnegative"
        ) from error
    if (
        not np.isfinite(onsets).all()
        or not np.isfinite(samples).all()
        or (onsets < 0).any()
        or (samples < 0).any()
        or not np.allclose(samples, np.rint(samples), rtol=0, atol=1e-9)
    ):
        raise ValueError(
            "Event onsets and integer samples must be finite and nonnegative"
        )
    annotation_onsets = np.asarray(raw.annotations.onset, dtype=float)
    if len(onsets) != len(annotation_onsets):
        raise ValueError("BIDS event count does not match raw annotations")
    sfreq = float(raw.info["sfreq"])
    onset_error = np.abs(onsets - annotation_onsets) * sfreq
    sample_error = np.abs(samples - annotation_onsets * sfreq)
    maximum = float(max(onset_error.max(initial=0), sample_error.max(initial=0)))
    if maximum > tolerance_samples + 1e-9:
        raise ValueError("BIDS events and raw annotations are misaligned")
    if maximum <= 1e-9:
        maximum = 0.0
    return {
        "rows": len(table),
        "maximum_alignment_error_samples": maximum,
        "counts": dict(sorted(Counter(descriptions).items())),
    }


def _validate_geometry(raw: mne.io.BaseRaw, controls: dict | None) -> dict:
    """Validate channel-name geometry against a declared standard montage."""
    if controls is None:
        return {
            "status": "coordinates_present_frame_not_independently_validated",
            "standard_template_mapping_validated": False,
        }
    required = {
        "standard_montage",
        "maximum_similarity_rmse_mm",
        "minimum_pairwise_distance_correlation",
    }
    missing = sorted(required - controls.keys())
    if missing:
        raise ValueError(f"Geometry controls are missing fields: {missing}")
    maximum_rmse = float(controls["maximum_similarity_rmse_mm"])
    minimum_correlation = float(controls["minimum_pairwise_distance_correlation"])
    if not np.isfinite(maximum_rmse) or maximum_rmse <= 0:
        raise ValueError("Geometry RMSE threshold must be finite and positive")
    if not np.isfinite(minimum_correlation) or not -1 <= minimum_correlation <= 1:
        raise ValueError("Geometry correlation threshold must be finite and bounded")

    montage_name = str(controls["standard_montage"])
    montage = mne.channels.make_standard_montage(montage_name)
    allow_subset = bool(controls.get("allow_montage_subset", False))
    if allow_subset:
        expected_names = [name for name in montage.ch_names if name in raw.ch_names]
        if raw.ch_names != expected_names:
            raise ValueError(
                "Raw channel names do not form an ordered standard-montage subset"
            )
    else:
        expected_names = montage.ch_names
        if raw.ch_names != expected_names:
            raise ValueError(
                "Raw channel names do not match the standard montage in order"
            )
    observed = np.asarray([channel["loc"][:3] for channel in raw.info["chs"]])
    positions = montage.get_positions()["ch_pos"]
    expected = np.asarray([positions[name] for name in expected_names])
    if not np.isfinite(observed).all() or not np.isfinite(expected).all():
        raise ValueError("Geometry comparison requires finite coordinates")

    observed_centered = observed - observed.mean(axis=0)
    expected_centered = expected - expected.mean(axis=0)
    left, singular_values, right = np.linalg.svd(
        observed_centered.T @ expected_centered
    )
    if np.linalg.det(left @ right) < 0:
        right[-1, :] *= -1
        singular_values[-1] *= -1
    rotation = left @ right
    denominator = float(np.sum(observed_centered**2))
    if denominator <= np.finfo(float).eps:
        raise ValueError("Observed geometry is degenerate")
    scale = float(np.sum(singular_values) / denominator)
    fitted = scale * observed_centered @ rotation + expected.mean(axis=0)
    residual_mm = np.linalg.norm(fitted - expected, axis=1) * 1000

    upper = np.triu_indices(len(observed), k=1)
    observed_distances = np.linalg.norm(
        observed[:, None, :] - observed[None, :, :], axis=2
    )[upper]
    expected_distances = np.linalg.norm(
        expected[:, None, :] - expected[None, :, :], axis=2
    )[upper]
    correlation = float(np.corrcoef(observed_distances, expected_distances)[0, 1])
    rmse_mm = float(np.sqrt(np.mean(residual_mm**2)))
    if not np.isfinite(correlation) or correlation < minimum_correlation:
        raise ValueError("Embedded geometry does not match the standard montage")
    if rmse_mm > maximum_rmse:
        raise ValueError("Embedded geometry exceeds the standard-montage RMSE limit")
    return {
        "status": "standard_template_layout_validated_individual_head_fit_unavailable",
        "standard_template_mapping_validated": True,
        "standard_montage": montage_name,
        "montage_subset": allow_subset,
        "matched_channels": len(observed),
        "similarity_scale": scale,
        "similarity_rmse_mm": rmse_mm,
        "maximum_residual_mm": float(residual_mm.max(initial=0)),
        "pairwise_distance_correlation": correlation,
        "thresholds": {
            "maximum_similarity_rmse_mm": maximum_rmse,
            "minimum_pairwise_distance_correlation": minimum_correlation,
        },
    }


def _validate_channels(
    raw: mne.io.BaseRaw, table: pd.DataFrame, profile: dict, metadata: dict
) -> dict:
    if table.columns.tolist() != ["name", "type", "units"]:
        raise ValueError("channels.tsv must contain exactly name, type, and units")
    names = table["name"].astype(str).tolist()
    if names != raw.ch_names:
        raise ValueError("channels.tsv names do not match raw channels in order")
    if table["name"].duplicated().any():
        raise ValueError("channels.tsv contains duplicate channel names")
    expected = profile["expected"]
    if len(names) != int(expected["eeg_channel_count"]):
        raise ValueError("Unexpected EEG channel count")
    if set(table["type"].astype(str)) != {"EEG"}:
        raise ValueError("Only EEG channels are supported by this profile")
    if set(table["units"].astype(str)) != {str(expected["channel_unit"])}:
        raise ValueError("Unexpected or mixed EEG channel units")
    if set(raw.get_channel_types()) != {"eeg"}:
        raise ValueError("MNE channel types do not match channels.tsv")
    if int(metadata["EEGChannelCount"]) != len(names):
        raise ValueError("EEGChannelCount does not match channels.tsv")
    positions = np.asarray([channel["loc"][:3] for channel in raw.info["chs"]])
    positioned = int(
        np.sum(
            np.isfinite(positions).all(axis=1) & (np.linalg.norm(positions, axis=1) > 0)
        )
    )
    if positioned < int(expected["minimum_positioned_channels"]):
        raise ValueError("Too few channels have usable digitized positions")
    geometry = _validate_geometry(raw, profile.get("geometry_validation"))
    reference_name = profile["reference_channel"]["name"]
    if reference_name not in raw.ch_names:
        raise ValueError("Declared reference channel is absent")
    reference_data = raw.get_data(picks=[reference_name])
    reference_flat = bool(np.ptp(reference_data) <= np.finfo(float).eps)
    if profile["reference_channel"]["must_be_flat"] and not reference_flat:
        raise ValueError("Declared online-reference channel is not flat")
    return {
        "eeg_channels": len(names),
        "channels_with_finite_embedded_coordinates": positioned,
        "geometry": geometry,
        "reference_channel_flat": reference_flat,
        "reference_policy": {
            "exclude_from_qc": bool(profile["reference_channel"]["exclude_from_qc"]),
            "exclude_from_reference_average": bool(
                profile["reference_channel"]["exclude_from_reference_average"]
            ),
            "post_reference_policy": profile["reference_channel"][
                "post_reference_policy"
            ],
        },
    }


def _validate_metadata(metadata: dict, description: dict, profile: dict) -> dict:
    expected = profile["expected"]
    checks = {
        "dataset_name": description.get("Name") == profile["dataset"]["name"],
        "snapshot": description.get("DatasetDOI")
        == (
            f"doi:10.18112/openneuro.{profile['dataset']['accession']}"
            f".v{profile['dataset']['snapshot']}"
        ),
        "task": metadata.get("TaskName") == profile["task"],
        "sampling_frequency": float(metadata.get("SamplingFrequency", -1))
        == float(expected["sampling_frequency_hz"]),
        "reference": metadata.get("EEGReference") == expected["reference"],
        "power_line_frequency": float(metadata.get("PowerLineFrequency", -1))
        == float(expected["power_line_frequency_hz"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("BIDS metadata gate failed: " + ", ".join(failed))
    return checks


def _validate_epoch_events(event_summary: dict, profile: dict) -> dict:
    counts = event_summary["counts"]
    epochs = profile["events"]["epoch_definitions"]
    result = {}
    for value, definition in sorted(epochs.items()):
        count = int(counts.get(value, 0))
        if count < int(definition["minimum_count"]):
            raise ValueError(f"Too few {value!r} events for epoching")
        window = [float(item) for item in definition["window_seconds"]]
        baseline = [float(item) for item in definition["baseline_seconds"]]
        if len(window) != 2 or len(baseline) != 2 or not window[0] < 0 < window[1]:
            raise ValueError(f"Invalid epoch definition for {value!r}")
        if not window[0] <= baseline[0] <= baseline[1] <= 0:
            raise ValueError(f"Invalid baseline definition for {value!r}")
        result[value] = {
            "count": count,
            "window_seconds": window,
            "baseline_seconds": baseline,
        }
        result[value]["purpose"] = definition["purpose"]
    excluded = profile["events"].get("excluded_values", {})
    excluded_present = {
        value: {"count": int(counts[value]), "reason": reason}
        for value, reason in sorted(excluded.items())
        if value in counts
    }
    context = profile["events"].get("context_values", {})
    context_present = {
        value: {"count": int(counts[value]), "reason": reason}
        for value, reason in sorted(context.items())
        if value in counts
    }
    known = set(epochs) | set(excluded) | set(context)
    unknown = sorted(set(counts) - known)
    if unknown:
        raise ValueError(
            "Observed events lack an analysis decision: " + ", ".join(unknown)
        )
    return {
        "epoch_definitions": result,
        "context_events": context_present,
        "excluded_events": excluded_present,
    }


def _amplitude_summary(raw: mne.io.BaseRaw, reference: str) -> dict:
    picks = [name for name in raw.ch_names if name != reference]
    data_uv = raw.get_data(picks=picks) * 1e6
    if not np.isfinite(data_uv).all():
        raise ValueError("EEG data contain non-finite values")
    centered = data_uv - np.median(data_uv, axis=1, keepdims=True)
    channel_sd = np.std(centered, axis=1)
    return {
        "method": (
            "MNE-scaled values after BIDS-declared uV source units; per-channel "
            "temporal median centering; descriptive only"
        ),
        "channels": len(picks),
        "channel_sd_uv_percentiles": {
            label: float(value)
            for label, value in zip(
                ("minimum", "p25", "median", "p75", "maximum"),
                np.percentile(channel_sd, [0, 25, 50, 75, 100]),
                strict=True,
            )
        },
    }


def run_inventory(inputs: BIDSEeglabInputs, profile_path: Path) -> tuple[dict, dict]:
    """Validate one recording and return its audit plus provenance core."""
    profile, profile_sha = _captured_json_sha(profile_path)
    initial_manifest = source_manifest()
    resolved = resolve_inputs(inputs)
    initial_inputs = input_identities(resolved)
    identities = {record["role"]: record for record in initial_inputs}
    description = json.loads(
        _capture(
            resolved.dataset_description,
            identities["dataset_description"]["sha256"],
        ).decode("utf-8-sig")
    )
    metadata = _merge_json(resolved.eeg_json, identities, "eeg_json")
    events_metadata = _merge_json(resolved.events_json, identities, "events_json")
    event_table = pd.read_csv(
        BytesIO(_capture(resolved.events_tsv, identities["events_tsv"]["sha256"])),
        sep="\t",
        keep_default_na=False,
    )
    channel_table = pd.read_csv(
        BytesIO(_capture(resolved.channels_tsv, identities["channels_tsv"]["sha256"])),
        sep="\t",
        keep_default_na=False,
    )
    raw = _read_raw(resolved.raw_set)
    recording = _recording_identity(resolved.raw_set, profile)

    metadata_checks = _validate_metadata(metadata, description, profile)
    channel_summary = _validate_channels(raw, channel_table, profile, metadata)
    event_summary = _event_alignment(
        raw,
        event_table,
        float(profile["events"]["alignment_tolerance_samples"]),
    )
    epoch_summary = _validate_epoch_events(event_summary, profile)
    documented = set(events_metadata.get("value", {}).get("Levels", {}))
    event_values = set(event_summary["counts"])
    explicitly_excluded = set(profile["events"].get("excluded_values", {}))
    undocumented = sorted(event_values - documented - explicitly_excluded)
    if undocumented:
        raise ValueError(
            "Events are neither documented nor explicitly excluded: "
            + ", ".join(undocumented)
        )

    final_inputs = input_identities(resolved)
    if final_inputs != initial_inputs:
        raise ValueError("Dataset inputs changed during inventory")
    if (
        sha256_file(profile_path) != profile_sha
        or source_manifest() != initial_manifest
    ):
        raise ValueError("Adapter controls changed during inventory")

    audit = {
        "schema_version": "1",
        "status": "pass",
        "scope": (
            "public BIDS/EEGLAB structural inventory licensing controlled "
            "non-spatial channel and temporal QC only; no interpolation, "
            "rereferencing, preprocessing, epoch inference, or publication"
        ),
        "dataset": profile["dataset"],
        "recording": recording,
        "task": profile["task"],
        "metadata_gates": metadata_checks,
        "channels": channel_summary,
        "events": {**event_summary, **epoch_summary},
        "amplitude_screen": _amplitude_summary(
            raw, profile["reference_channel"]["name"]
        ),
        "classification": "controlled_derived",
        "publication_allowed": False,
    }
    core = {
        "schema_version": "1",
        "inputs": initial_inputs,
        "controls": {
            "profile_sha256": profile_sha,
            "source_manifest": initial_manifest,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions(),
            "dependency_files": {
                name: sha256_file(Path(__file__).resolve().parents[2] / name)
                for name in ("requirements.txt", "constraints-ci.txt")
            },
        },
        "audit_sha256": canonical_sha256(audit),
        "classification": "controlled_derived",
        "publication_allowed": False,
    }
    core["core_sha256"] = canonical_sha256(core)
    return audit, core


def publish_inventory(
    inputs: BIDSEeglabInputs, profile_path: Path, output: Path
) -> Path:
    """Publish one exact-set-verified inventory atomically to a new directory."""
    output = Path(output).expanduser().absolute()
    dataset_root = (
        Path(inputs.dataset_root).expanduser().absolute().resolve(strict=True)
    )
    nearest = output.parent
    while not nearest.exists():
        nearest = nearest.parent
    if nearest.resolve(strict=True).is_relative_to(dataset_root):
        raise ValueError("Output must be outside the source dataset")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        audit, core = run_inventory(inputs, profile_path)
        (temporary / "audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_provenance(temporary, core)
        verify_provenance(temporary)
        resolved = resolve_inputs(inputs)
        if input_identities(resolved) != core["inputs"]:
            raise ValueError("Dataset inputs changed before publication")
        if sha256_file(profile_path) != core["controls"]["profile_sha256"]:
            raise ValueError("Adapter profile changed before publication")
        if source_manifest() != core["controls"]["source_manifest"]:
            raise ValueError("Executable source changed before publication")
        temporary.replace(output)
        try:
            verify_provenance(output)
        except Exception:
            shutil.rmtree(output, ignore_errors=True)
            raise
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output
