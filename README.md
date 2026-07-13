# Dense-EEG stop-signal pipeline

[![CI](https://github.com/viranovskaya/dense-eeg-stop-signal-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/viranovskaya/dense-eeg-stop-signal-pipeline/actions/workflows/ci.yml)

A Python/MNE workflow for 129-channel BrainVision EEG recorded during a stop-signal task. This project started from real recordings, short preprocessing notes, old EEGLAB files, and marker sequences whose meaning had to be reconstructed before the data could be analysed again.

My first goal was to recover what had actually been done to the data. I then wrote a new pipeline in which the event rules, bad-channel decisions, reference, and output files can be checked for every participant.

## Implemented and checked

- BrainVision import, including a fix for malformed dates in old marker files;
- correct ECG/EOG channel types and a standard 10-05 montage;
- QC based on amplitude, flat segments, PSD, and 50 Hz noise;
- reconstruction of go, successful-stop, failed-stop, and unresolved trials;
- reviewed bad-channel lists followed by EEG-channel interpolation;
- 1–40 Hz filtering and EEG-only average reference;
- continuous, go-locked, and stop-locked files in FIF and EEGLAB `.set` formats;
- low-pass C3/C4 ERP summaries for the first motor-response checks.

I ran the complete QC and preprocessing workflow on 10 recordings: 121.7 minutes of 129-channel EEG sampled at 1000 Hz. All 10 produced continuous, go-locked, and stop-locked files in both formats. The recordings and participant reports are not included in the repository.

The event mapping is covered by synthetic regression tests and was cross-checked against the original marker sequences and preserved condition datasets. Automated bad-channel flags were reviewed across five recording windows before persistent channels were approved for interpolation.

## Planned, not yet claimed as results

- ICA fitting and a saved table of component decisions;
- systematic time-frequency and scalp-topography summaries;
- template-based exploratory source localization.

These steps are intentionally listed as planned. The public repository should not read as if ICA cleaning or source localization has already been completed.

## Recovered event logic

- `S17`, `S18`: go-stimulus variants;
- `S6`: correct go response;
- `S4`: incorrect or too-slow go response;
- `S1`, `S2`: stimulus variants on stop trials;
- `S19`: stop signal;
- `S5` after `S19`: failed stop;
- no `S5` before the next trial: successful stop;
- `S7`: rare unresolved sequence, excluded from confirmatory analysis.

I describe where this mapping came from, and what is still uncertain, in [`docs/recovered_protocol.md`](docs/recovered_protocol.md).

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

The QC stage writes CSV/JSON summaries, figures, and a Markdown report. It only suggests possible bad channels. I check them visually and record the final decision in a manifest. ECG and EOG are not interpolated or included in the EEG reference.

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

The tests cover marker normalisation, trial reconstruction from a synthetic marker fixture, the rule that `S7` sequences remain unclassified, and the compatibility path for malformed BrainVision marker dates. They also run automatically on every pull request.

## What the available data allow

The public repository contains code and tests, not participant recordings. The files available to me also do not include group labels, individual MRI, or digitised electrode positions. Therefore I cannot reconstruct the group comparison, and any later source localisation would have to use a template and remain exploratory.

## Citation and license

Citation information is in [`CITATION.cff`](CITATION.cff). The code is released under the [MIT License](LICENSE).
