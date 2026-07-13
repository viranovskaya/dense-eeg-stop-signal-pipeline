# Dense-EEG stop-signal pipeline

[![CI](https://github.com/viranovskaya/dense-eeg-stop-signal-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/viranovskaya/dense-eeg-stop-signal-pipeline/actions/workflows/ci.yml)

A reproducible Python/MNE workflow for quality control, preprocessing, and event reconstruction in 129-channel BrainVision EEG recorded during a stop-signal task.

## What works now

### Implemented

- non-destructive BrainVision import, including a compatibility path for malformed legacy measurement dates;
- ECG/EOG typing and standard 10-05 montage assignment;
- dataset-level QC with channel-amplitude, flatness, PSD, and 50 Hz metrics;
- reconstruction of go, successful-stop, failed-stop, and unresolved trials;
- explicit bad-channel manifests followed by EEG-only interpolation;
- 1–40 Hz filtering and EEG-only common-average reference;
- continuous, go-epoch, and stop-epoch export to FIF and EEGLAB `.set`;
- C3/C4 low-pass ERP summaries for go- and stop-locked epochs.

### Validated locally

The QC and preprocessing commands have run end to end on 10 complete recordings (121.7 minutes of 129-channel EEG at 1000 Hz). All 10 produced continuous, go-epoch, and stop-epoch files in both FIF and EEGLAB formats. The public repository contains code and tests only; participant data and participant-level reports remain local.

The event mapping was cross-checked against the original marker sequences and preserved condition datasets. Automated bad-channel flags were reviewed across five recording windows before persistent channels were approved for interpolation.

### Not implemented yet

- ICA fitting and component rejection with an explicit decision table;
- systematic time-frequency and scalp-topography summaries;
- template-based exploratory source localization.

## My contribution

I reconstructed the task and preprocessing history from dataset notes, BrainVision markers, saved EEGLAB histories, and condition files. I then translated that reconstruction into the Python/MNE pipeline in this repository, added dataset-level QC, made channel and reference decisions explicit, and tested the workflow on the available recordings.

## Recovered event logic

- `S17`, `S18`: go-stimulus variants;
- `S6`: correct go response;
- `S4`: incorrect or too-slow go response;
- `S1`, `S2`: stimulus variants on stop trials;
- `S19`: stop signal;
- `S5` after `S19`: failed stop;
- no `S5` before the next trial: successful stop;
- `S7`: rare unresolved sequence, excluded from confirmatory analysis.

The evidence and unresolved points are documented in [`docs/recovered_protocol.md`](docs/recovered_protocol.md).

## Run quality control

For one recording:

```bash
python scripts/run_qc.py \
  --vhdr /path/to/sub-001_task-stop.vhdr \
  --output results/sub-001
```

For all `.vhdr` files in a directory:

```bash
python scripts/run_dataset_qc.py \
  --input-dir /path/to/brainvision-folder \
  --output results/dataset-qc
```

The QC stage writes CSV/JSON summaries, figures, and a Markdown report. Candidate bad channels are never removed automatically: they must be confirmed visually and recorded in a manifest. ECG/EOG are not interpolated or included in the EEG reference.

## Run preprocessing

For one participant:

```bash
python scripts/run_preprocess.py \
  --vhdr /path/to/sub-001_task-stop.vhdr \
  --participant-id 001 \
  --bad-channels CHAN1,CHAN2 \
  --output results/processed/sub-001
```

For a complete directory with reviewed channel decisions:

```bash
python scripts/run_dataset_preprocess.py \
  --input-dir /path/to/brainvision-folder \
  --bad-channel-manifest results/dataset-qc/bad_channels_manifest.csv \
  --output results/processed
```

A MATLAB/EEGLAB implementation of the recovered workflow is available in [`matlab/preprocess_eeglab.m`](matlab/preprocess_eeglab.m).

## Test

```bash
python -m unittest discover -s tests -v
```

The tests exercise marker normalization and trial reconstruction, including the rule that `S7` sequences remain unclassified. The same suite runs automatically on every pull request.

## Next milestone

The next analysis step is ICA on continuous filtered EEG. Each rejected component will be stored with its component number, decision, reason, and reviewer before subtraction. ERP/time-frequency summaries will follow only after those decisions are reproducible.

## Data and interpretation boundary

Raw and processed participant files are excluded by `.gitignore`. The available files do not include group labels, individual MRI, or digitized electrode positions, so group comparisons cannot be reconstructed and source localization can only be template-based and exploratory.

## Citation and license

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The code is released under the [MIT License](LICENSE); participant data are not distributed under this license.
