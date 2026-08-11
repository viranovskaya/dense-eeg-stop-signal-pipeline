# Dense-EEG stop-signal pipeline

[![CI](https://github.com/viranovskaya/dense-eeg-stop-signal-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/viranovskaya/dense-eeg-stop-signal-pipeline/actions/workflows/ci.yml)

## Purpose

This repository reconstructs the event logic and preprocessing history of
129-channel BrainVision EEG recorded during a stop-signal task. The main aim is
to make each trial classification and each new processing decision traceable.

## Practical problem

The recordings were accompanied by short notes, old EEGLAB files, and marker
sequences, but not a complete record of event mapping, bad-channel decisions,
rereferencing, or component rejection. I separate evidence recovered from the
old files from decisions made in the new workflow.

## What I implemented

I implemented:

- BrainVision import, including a compatibility path for malformed marker dates;
- ECG/EOG typing and a standard 10-05 montage;
- amplitude, flat-segment, PSD, and 50 Hz QC summaries;
- a complete non-overlapping temporal scan that keeps window-level evidence
  for intermittent channel problems;
- reconstruction of go, successful-stop, failed-stop, and unresolved trials;
- explicit `classified`, `inferred`, `ambiguous`, `incomplete`, and `invalid`
  trial states;
- qualitative rule-confidence labels carried from the executable event
  codebook into each trial row; these are not calibrated probabilities;
- trial and epoch accounting that keeps every detected trial start visible;
- explicit reviewed bad-channel manifests before EEG interpolation;
- 1--40 Hz filtering and EEG-only average reference;
- continuous, go-locked, and stop-locked FIF and EEGLAB exports;
- targeted before/after screening metrics;
- run provenance with input, configuration, source, decision, and output hashes;
- trial-to-epoch lineage tables that retain classification and drop reasons;
- low-pass C3/C4 ERP summaries as an initial signal check.

The recovered marker logic is documented in
[`docs/recovered_protocol.md`](docs/recovered_protocol.md). The scientific scope
and evidence boundary are in [`docs/methods_scope.md`](docs/methods_scope.md).

## Data and sample

The methods-evaluation workflow was rerun in a controlled local environment on
10 private recordings: 121.7 minutes of 129-channel EEG sampled at 1000 Hz.
All 10 recordings produced provenance-verified participant outputs and a
dataset-level aggregate. Participant data, channel decisions, and generated
reports are not tracked in this repository.

The public test data are a synthetic marker fixture. The available research material has no group labels, individual MRI, or digitized electrode positions.

## Private evaluation evidence

The controlled methods rerun produced continuous, go-locked, and stop-locked
FIF outputs for all 10 recordings. EEGLAB export remains supported by the code
and synthetic tests, but was disabled for this private rerun. The preserved
condition files for one recording provide the strongest direct check of the
event mapping. The remaining recordings support sequence-level checks, but
they are not an independent ground truth.

A failed stop is directly marked by a post-stop response. A successful stop is
instead inferred when no response marker occurs before the next trial begins.
The output keeps that distinction explicit.

Synthetic regression tests cover configuration validation, marker
normalization, complete and truncated trials, conflicting or unknown markers,
malformed BrainVision dates, temporal integrity limits, epoch lineage,
provenance, preprocessing order,
export contracts, and full recording coverage by the temporal QC scan.

## Reproducibility

[`config/analysis.json`](config/analysis.json) and
[`config/event_codebook.csv`](config/event_codebook.csv) are executable inputs,
not duplicated documentation. A run stops when either file is incomplete or
contradictory.

Dataset preprocessing requires a reviewed bad-channel table with a reason,
reviewer, date, and inspected windows for each decision. A synthetic template
is provided in
[`config/bad_channel_manifest_example.csv`](config/bad_channel_manifest_example.csv).
If review finds no persistent bad channels, the manifest records one explicit
participant-level `none` decision instead of silently omitting that recording.
The resulting `provenance.json` records SHA-256 digests without exposing source
file names or absolute paths. It includes a deterministic manifest of the
executable source, so a dirty working tree is not identified only by a Boolean.
Single-recording output is assembled in a temporary directory and moved into
place only after completion. Dataset commands require a new output path and do
not silently resume prior runs.

Generated QC and participant reports are private derived outputs. They may
contain pseudonymous IDs, recording summaries, and figures and must be reviewed
before sharing or publication.

CI installs the declared dependency ranges on Python 3.12, runs the synthetic unit suite, and compiles Python sources. Real participant-level reproduction requires separately controlled source files and the reviewed bad-channel manifest; those inputs are not public.

## Limitations

- Exact original bad-channel and ICA-rejection decisions cannot be recovered from the preserved history.
- ICA fitting and an explicit component-decision table are not yet implemented as completed results.
- Group comparisons cannot be reconstructed without group labels.
- Individual source localization is out of scope because MRI and digitized geometry are unavailable.
- The current public fixture is synthetic; an independent public stop-signal example has not yet been integrated.
- Window-level robust-z flags are conservative review prompts, not validated
  universal thresholds or automatic interpolation decisions. Common-mode
  artifacts may not appear as spatial outliers.
- A 20-second window can dilute a short artifact. The reported fraction is the
  fraction of affected windows, not the fraction of contaminated samples.
  Results can also depend on window alignment, and filtered excursions may
  include filter ringing; raw and filtered evidence are therefore retained.
- The dependency file gives supported ranges; an exact cross-machine environment for the private 10-recording run is not yet published.

## Installation and run

```bash
git clone https://github.com/viranovskaya/dense-eeg-stop-signal-pipeline.git
cd dense-eeg-stop-signal-pipeline
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Quality control for one recording:

```bash
python scripts/run_qc.py \
  --vhdr /path/to/sub-001_task-stop.vhdr \
  --output results/sub-001
```

Preprocessing after manual bad-channel review:

```bash
python scripts/run_preprocess.py \
  --vhdr /path/to/sub-001_task-stop.vhdr \
  --participant-id 001 \
  --bad-channels CHAN1,CHAN2 \
  --output results/processed/sub-001
```

The comma-separated `--bad-channels` option is useful for an exploratory single
run, but its provenance record is deliberately marked incomplete. The
dataset-level methods run uses the full review manifest:

```bash
python scripts/run_dataset_preprocess.py \
  --input-dir /path/to/brainvision \
  --bad-channel-manifest /path/to/bad_channel_decisions.csv \
  --output results/processed
```

Dataset-level commands are available in `scripts/run_dataset_qc.py` and
`scripts/run_dataset_preprocess.py`. The MATLAB script in
[`matlab/preprocess_eeglab.m`](matlab/preprocess_eeglab.m) is a historical
reference only; it is not equivalent to the current Python event and provenance
logic.

The dataset preprocessing command also writes a participant table, an aggregate
JSON report, and one before/after figure. These retain trial-status totals,
epoch losses and their reasons, decision completeness, interpolated-channel
counts and comparable post-filter amplitude and flatness summaries. Raw 50 Hz
line-noise screening remains part of raw QC, but is not used as a before/after
metric after the 1--40 Hz filter.

## Citation

Use [`CITATION.cff`](CITATION.cff). Versioned archives are available on the
[GitHub Releases](https://github.com/viranovskaya/dense-eeg-stop-signal-pipeline/releases)
page, and the code is released under the [MIT License](LICENSE).

## Current status

- **Implemented:** executable event definitions, conservative trial reconstruction, reconciliation, reviewed interpolation, filtering, rereferencing, epoching, provenance, and dual-format export.
- **Tested:** synthetic marker, configuration, provenance, epoch-accounting, and preprocessing contracts in CI.
- **Evaluated:** controlled methods run on 10 private recordings, with verified participant and dataset provenance.
- **Next:** add a synthetic corruption benchmark that measures both artifact detection and signal preservation.
- **Later:** add one public 128-channel stop-signal example and an explicit ICA component-decision table.
- **Not validated:** group analysis, exact original ICA choices, source localization, or generalization beyond the evaluated recordings.
