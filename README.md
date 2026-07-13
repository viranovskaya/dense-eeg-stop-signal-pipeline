# Dense-EEG stop-signal pipeline

Reproducible quality control and preprocessing scaffold for 129-channel BrainVision EEG recorded during a stop-signal task.

## Status

The repository currently implements a non-destructive raw-data QC pilot and reconstructs the task event logic from the original BrainVision markers and a preserved EEGLAB history. Human-participant data are not included.

Verified event logic:

- `S17`, `S18`: go-stimulus variants;
- `S6`: correct go response;
- `S4`: incorrect or too-slow go response;
- `S1`, `S2`: stimulus variants on stop trials;
- `S19`: stop signal;
- `S5`: response after the stop signal, therefore failed/bad stop;
- a stop trial without `S5` before the next trial is a successful/good stop;
- `S7`: rare unresolved event, excluded from confirmatory analysis.

The recovered protocol and its reproducibility caveats are documented in [`docs/recovered_protocol.md`](docs/recovered_protocol.md).

## Safety

Raw and processed participant files are excluded by `.gitignore`. Do not commit `.eeg`, `.vhdr`, `.vmrk`, `.set`, `.fif`, `.mat`, participant tables, or participant-level QC reports without explicit data-owner and ethics approval.

## Run a QC pilot

```bash
python scripts/run_qc.py \
  --vhdr /path/to/sub-001_task-stop.vhdr \
  --output results/sub-001
```

The QC command:

1. reads BrainVision data without modifying it;
2. marks ECG and EOG as non-EEG channels;
3. attaches the standard 10-05 montage for visualization;
4. reconstructs go, successful-stop, and failed-stop trials;
5. computes representative channel-amplitude and spectral metrics;
6. flags candidate bad channels for visual confirmation;
7. writes CSV/JSON summaries and a Markdown report.

To run the same checks for every `.vhdr` file in one directory:

```bash
python scripts/run_dataset_qc.py \
  --input-dir /path/to/brainvision-folder \
  --output results/dataset-qc
```

Candidate bad channels are **not** removed automatically. After visual confirmation they should be marked and interpolated from neighboring EEG electrodes. ECG/EOG must not be interpolated and are excluded from the average reference and EEG ICA.

To create a processed continuous file and go/stop epochs for one participant:

```bash
python scripts/run_preprocess.py \
  --vhdr /path/to/sub-001_task-stop.vhdr \
  --participant-id 001 \
  --bad-channels CHAN1,CHAN2 \
  --output results/processed/sub-001
```

The command writes local FIF and EEGLAB `.set` versions. Processed participant data remain ignored by Git. A MATLAB/EEGLAB implementation is also provided in [`matlab/preprocess_eeglab.m`](matlab/preprocess_eeglab.m).

After QC decisions have been saved in a manifest, the same operation can be applied to the complete directory:

```bash
python scripts/run_dataset_preprocess.py \
  --input-dir /path/to/brainvision-folder \
  --bad-channel-manifest results/dataset-qc/bad_channels_manifest.csv \
  --output results/processed
```

## Planned preprocessing

- assign channel types and standard montage;
- inspect and mark bad EEG channels;
- 1–40 Hz filtering;
- average reference using EEG channels only;
- interpolate confirmed bad EEG channels;
- epoch go trials around `S17/S18` (−1.5 to 3.0 s);
- epoch stop trials around `S19` (−2.0 to 2.0 s);
- split successful and failed stop trials using the presence of `S5`;
- fit ICA on continuous filtered EEG, inspect components, and record every rejected component explicitly;
- create C3/C4 and premotor/motor ERP and time-frequency summaries;
- treat template-based source localization as exploratory.

## Reproducibility boundary

The available files do not encode age group, cultural/language group, individual MRI, or digitized electrode positions. Those factors cannot be reconstructed or used for confirmatory group inference. Template-based source results must be presented as approximate and exploratory.
