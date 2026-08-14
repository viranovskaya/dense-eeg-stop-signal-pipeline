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
- rank-aware extended Infomax ICA with a separate private review package;
- a solution-bound `keep` or `exclude` decision for every ICA component;
- 1--40 Hz filtering and EEG-only average reference;
- continuous, go-locked, and stop-locked FIF and EEGLAB exports;
- targeted before/after screening metrics;
- run provenance with input, configuration, source, decision, and output hashes;
- trial-to-epoch lineage tables that retain classification and drop reasons;
- low-pass C3/C4 ERP summaries as an initial signal check;
- a public 127-channel corruption benchmark with known channel-window truth,
  held-out seeds, BrainVision round-trip checks, and separate detection and
  signal-preservation measures.

The recovered marker logic is documented in
[`docs/recovered_protocol.md`](docs/recovered_protocol.md). The scientific scope
and evidence boundary are in [`docs/methods_scope.md`](docs/methods_scope.md).
The component-review order and decision contract are described in
[`docs/ica_workflow.md`](docs/ica_workflow.md).
The public ICA integration check is reported in
[`docs/ica_validation.md`](docs/ica_validation.md).

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

## Public synthetic benchmark

[`docs/synthetic_benchmark.md`](docs/synthetic_benchmark.md) describes a fully
public 127-channel benchmark with EOG, ECG, complete go/stop marker sequences,
five declared corruption families, and a known C3/C4 task signal. The benchmark
keeps calibration, seed-level holdout and stress seeds separate and writes an
exact-set provenance record before publishing a new output directory. The
holdout uses new random seeds under the same fixed corruption scenario; it is
not a test of new artifact conditions or external generalization.

Across the three predeclared seed-level holdout recordings, the primary
temporal screen reached precision, recall and F1 of 1.00 for 13 positive cases
among 1,135 primary channel-window pairs per recording. The all-temporal set
contained 1,136 pairs including the separate stress case. The separate raw-PSD
50 Hz detector also reached F1 1.00. The 100-µV, 0.5-second electrode-pop stress
case was not detected after dilution inside a 20-second window.

After the declared oracle decisions, the largest C3/C4 peak-amplitude error was
2.48 µV and the peak-latency error was 0 ms. Across 1,524 channel-band values,
the 1,464 unaffected values had a median absolute error of 0.012 dB, a 95th
percentile of 0.092 dB and a maximum of 0.141 dB. Errors were much larger for
the 24 oracle-interpolated values and the 36 values from other corrupted
channels. The complete center, upper-tail and maximum statistics are retained
in the [machine-readable held-out summary](docs/benchmark_results/heldout_seed_summary.json).
These are controlled synthetic software results, not evidence that unknown EEG
signals can be reconstructed or estimates of performance on participant data.

## Public ICA integration check

The explicit ICA workflow was also run end to end on one deterministic public
synthetic fixture: 127 EEG channels plus EOG and ECG, 180 seconds at 250 Hz.
Extended Infomax met the MNE small-angle stopping rule (`n_small_angle=20`)
after 530 iterations and estimated 126 components. Review identified one ocular
and one cardiac component; all 126 components then received an explicit
decision, and only those two were excluded.

Across the 127 EEG channels, the maximum absolute EOG correlation fell from
0.295 to 0.028 and the maximum absolute ECG correlation from 0.176 to 0.049.
The largest C3/C4 task-peak change was 0.23 µV with no peak-latency shift. The
median absolute change across 508 channel-band values was 0.082 dB; the 95th
percentile was 0.645 dB and the maximum was 0.994 dB. All 34 reconstructed
trial starts were accounted for, with 17 go and 17 stop epochs retained and no
epoch drops.

This is an integration check on one fixed synthetic recording. It shows that
the review, decision-binding, application and reporting path works as intended;
it does not validate automatic component classification or performance on
participant EEG. Exact values and environment versions are in the
[machine-readable summary](docs/benchmark_results/ica_seed_4401_summary.json).

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
- The new ICA workflow records current decisions; it does not reconstruct the
  undocumented component choices made in the historical EEGLAB analysis.
- Group comparisons cannot be reconstructed without group labels.
- Individual source localization is out of scope because MRI and digitized geometry are unavailable.
- The current public fixture is synthetic; an independent public stop-signal example has not yet been integrated.
- Synthetic benchmark results depend on the declared generator, corruption
  amplitudes and fixed seeds and do not establish generalization to real EEG.
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

ICA is optional and never removes components automatically. First create a
public fixture when reproducing the integration example:

```bash
python scripts/create_ica_validation_fixture.py \
  --output /new/path/public-ica-fixture
```

For a real recording, start directly with a private review package from the
same reviewed bad-channel record:

```bash
python scripts/run_ica_review.py \
  --vhdr /path/to/sub-001_task-stop.vhdr \
  --participant 001 \
  --bad-channel-manifest /path/to/bad_channel_decisions.csv \
  --output /private/path/ica-review/sub-001
```

Review the topographies, time courses, spectra and EOG/ECG correlations. Copy
`ica_decision_template.csv` outside the immutable review package, then complete
every row with an explicit `keep` or `exclude` decision. The final preprocessing
call binds that table to the exact ICA solution by SHA-256:

```bash
python scripts/run_preprocess.py \
  --vhdr /path/to/sub-001_task-stop.vhdr \
  --participant-id 001 \
  --bad-channel-manifest /path/to/bad_channel_decisions.csv \
  --ica-solution /private/path/ica-review/sub-001/ica_solution.fif \
  --ica-decisions /private/path/ica_decisions.csv \
  --output /private/path/processed/sub-001
```

Dataset preprocessing accepts the same decision table through
`--ica-decision-manifest` and a directory of participant review packages
through `--ica-review-root`.

After a public fixture has been reviewed and processed, reproduce the compact
validation summary with:

```bash
python scripts/summarize_ica_validation.py \
  --vhdr /new/path/public-ica-fixture/synthetic_ica.vhdr \
  --review-package /private/path/ica-review/synthetic \
  --decisions /private/path/ica_decisions.csv \
  --processed-output /private/path/processed/synthetic \
  --fixture-seed 4401 \
  --output /new/path/ica_validation_summary.json
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

- **Implemented:** executable event definitions, conservative trial reconstruction, reconciliation, reviewed interpolation, filtering, rereferencing, explicit ICA review and decisions, epoching, provenance, and dual-format export.
- **Tested:** synthetic marker, configuration, provenance, epoch-accounting, and preprocessing contracts in CI.
- **Evaluated:** controlled methods run on 10 private recordings, with verified participant and dataset provenance.
- **Benchmark:** known-truth detection, BrainVision round trip, trial accounting,
  band-power preservation and C3/C4 task-signal preservation are implemented.
- **Next:** define a provenance-bound segment-review input before any private
  ICA rerun, then add one independent public stop-signal example.
- **Not validated:** group analysis, exact original ICA choices, source localization, or generalization beyond the evaluated recordings.
