# Dense-EEG stop-signal pipeline

[![CI](https://github.com/viranovskaya/dense-eeg-stop-signal-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/viranovskaya/dense-eeg-stop-signal-pipeline/actions/workflows/ci.yml)

## Purpose

This repository reconstructs and makes auditable a preprocessing workflow for 129-channel BrainVision EEG recorded during a stop-signal task.

## Practical problem

The available recordings were accompanied by short notes, old EEGLAB files, and marker sequences, but not a complete reproducible record of event mapping, bad-channel decisions, rereferencing, or component rejection. The project separates what can be recovered from what must be decided explicitly in a new run.

## What I implemented

I implemented:

- BrainVision import, including a compatibility path for malformed marker dates;
- ECG/EOG typing and a standard 10-05 montage;
- amplitude, flat-segment, PSD, and 50 Hz QC summaries;
- reconstruction of go, successful-stop, failed-stop, and unresolved trials;
- explicit reviewed bad-channel manifests before EEG interpolation;
- 1--40 Hz filtering and EEG-only average reference;
- continuous, go-locked, and stop-locked FIF and EEGLAB exports;
- low-pass C3/C4 ERP summaries for initial motor-response checks.

The recovered marker logic and evidence are documented in [`docs/recovered_protocol.md`](docs/recovered_protocol.md).

## Data and sample

I evaluated the complete QC and preprocessing workflow on 10 private recordings: 121.7 minutes of 129-channel EEG sampled at 1000 Hz. Participant recordings and reports are not tracked in this repository.

The public test data are a synthetic marker fixture. The available research material has no group labels, individual MRI, or digitized electrode positions.

## Validated outputs

All 10 evaluated recordings produced continuous, go-locked, and stop-locked datasets in FIF and EEGLAB formats. Event reconstruction was cross-checked against original marker sequences and preserved condition datasets. Automated bad-channel candidates were reviewed across five recording windows before persistent channels were approved for interpolation.

Synthetic regression tests cover marker normalization, trial reconstruction, unresolved `S7` sequences, malformed BrainVision dates, preprocessing order, and export contracts.

## Reproducibility

Configuration lives in [`config/analysis.json`](config/analysis.json) and [`config/event_codebook.csv`](config/event_codebook.csv). The Python and MATLAB implementations preserve the recovered event definitions, while the Python workflow records reviewed bad channels explicitly.

CI installs the declared dependency ranges on Python 3.12, runs the synthetic unit suite, and compiles Python sources. Real participant-level reproduction requires separately controlled source files and the reviewed bad-channel manifest; those inputs are not public.

## Limitations

- Exact original bad-channel and ICA-rejection decisions cannot be recovered from the preserved history.
- ICA fitting and an explicit component-decision table are not yet implemented as completed results.
- Group comparisons cannot be reconstructed without group labels.
- Source localization would be template-based and exploratory because individual MRI and digitized geometry are unavailable.
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

Dataset-level commands are available in `scripts/run_dataset_qc.py` and `scripts/run_dataset_preprocess.py`. A recovered EEGLAB implementation is in [`matlab/preprocess_eeglab.m`](matlab/preprocess_eeglab.m).

## Citation

Use [`CITATION.cff`](CITATION.cff). GitHub release `v0.1.0` is available, and the code is released under the [MIT License](LICENSE).

## Current status

- **Implemented:** QC, event reconstruction, reviewed interpolation, filtering, rereferencing, epoching, and dual-format export.
- **Tested:** synthetic marker and preprocessing contracts in CI.
- **Evaluated:** complete workflow on 10 private recordings.
- **Planned:** explicit ICA component decisions, time-frequency summaries, and exploratory template source localization.
- **Not yet validated:** group analysis, exact original ICA choices, individual source localization, or generalization beyond the evaluated recordings.
