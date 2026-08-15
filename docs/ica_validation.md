# Public ICA integration check

## Question

This check asks whether the explicit ICA workflow can be completed from public
BrainVision input through review, component decisions, preprocessing and a
bounded before/after summary. It does not ask whether a correlation threshold
can classify components automatically.

## Fixture

The fixture is generated from seed 4401 using the public synthetic recording
model in this repository. It contains 127 EEG channels, one EOG channel, one
ECG channel, 77 task markers and 180 seconds of data sampled at 250 Hz. The
generator exports BrainVision and verifies the channel order, samples, signal
values and marker order after rereading the files.

Two independent fixture-generation runs produced byte-identical output
directories in the pinned test environment.

## Review and decisions

Extended Infomax met the MNE small-angle stopping rule
(`n_small_angle=20`) after 530 iterations and estimated 126 components from the
reviewed good EEG rank. The topography overview and detailed candidate plots
supported two exclusions:

- component 2: EOG correlation 0.869, frontal topography and repeated
  blink-like segments;
- component 3: ECG correlation 0.742, cardiac-rate spectrum and repeated
  cardiac waveform.

Every component received an explicit `keep` or `exclude` decision. Component
numbers are zero-based. The decision table was bound to the exact ICA solution
SHA-256. The software did not make or apply an automatic exclusion.

## Before/after checks

Across all 127 EEG channels:

| Check | Before | After |
| --- | ---: | ---: |
| Maximum absolute EEG-EOG correlation | 0.295 | 0.028 |
| Median absolute EEG-EOG correlation | 0.130 | 0.006 |
| Maximum absolute EEG-ECG correlation | 0.176 | 0.049 |
| Median absolute EEG-ECG correlation | 0.074 | 0.012 |

The largest C3/C4 task-peak amplitude change was 0.230 µV and the largest
peak-latency change was 0 ms. Across 508 channel-band values, the median
absolute band-power change was 0.082 dB, the 95th percentile was 0.645 dB and
the maximum was 0.994 dB.

Event accounting remained complete: 77 markers, 34 trial starts, 17 retained
go epochs, 17 retained stop epochs and no dropped epochs.

## Interpretation

The selected components carried strong auxiliary-channel relationships, and
their reviewed removal reduced those relationships while leaving the declared
C3/C4 peak timing unchanged and producing small-to-moderate spectral changes
in this fixture. These checks are deliberately reported together: lower
EOG/ECG correlation alone is not enough to claim successful cleaning.

This is one fixed synthetic integration fixture. It is not participant-level
validation, an estimate of sensitivity or specificity, or evidence that the
same decisions should be made in another recording. The exact machine-readable
result is in
[`benchmark_results/ica_seed_4401_summary.json`](benchmark_results/ica_seed_4401_summary.json).
The tracked [tidy aggregate table](paper_assets/tables/synthetic_ica_validation.csv) is
generated from that summary with `scripts/build_paper_assets.py`.
