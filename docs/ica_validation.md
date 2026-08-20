# Public ICA integration check

## Question

This check asks whether the explicit ICA workflow can be completed from public
BrainVision input through review, component decisions, preprocessing and a
bounded matched-control summary. The control output uses the same input,
filter, reference, interval decisions, ICA solution and interpolation decisions
but retains every ICA component. It does not ask whether a correlation
threshold can classify components automatically.

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

## Matched-control checks

Across all 127 EEG channels:

| Check | All components retained | Reviewed exclusions |
| --- | ---: | ---: |
| Maximum absolute EEG-EOG correlation | 0.358 | 0.071 |
| Median absolute EEG-EOG correlation | 0.161 | 0.017 |
| Maximum absolute EEG-ECG correlation | 0.158 | 0.043 |
| Median absolute EEG-ECG correlation | 0.068 | 0.011 |

The task-peak and band-power values below compare those two otherwise matched
outputs. They therefore isolate the effect of the reviewed component exclusions
from the separate 1--40 Hz ICA-fit filter and 0.2--30 Hz ERP filter.

The largest C3/C4 task-peak change was 0.200 µV, with no peak-latency shift.
Across 508 channel-band values, the median absolute band-power change was
0.082 dB, the 95th percentile was 0.649 dB and the maximum was 0.996 dB.

Event accounting remained complete: 77 markers, 34 trial starts, 17 retained
go epochs, 17 retained stop epochs and no dropped epochs.

## Interpretation

The selected components carried strong auxiliary-channel relationships. These
checks are deliberately reported together: lower EOG/ECG correlation alone is
not enough to claim successful cleaning, and preservation is evaluated against
the matched all-components-retained control rather than against the differently
filtered ICA-fit input.

This is one fixed synthetic integration fixture. It is not participant-level
validation, an estimate of sensitivity or specificity, or evidence that the
same decisions should be made in another recording. The exact machine-readable
result is in
[`benchmark_results/ica_seed_4401_summary.json`](benchmark_results/ica_seed_4401_summary.json).
