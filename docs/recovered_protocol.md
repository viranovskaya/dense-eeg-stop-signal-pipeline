# Recovered EEGLAB protocol and event logic

## Evidence

The protocol was reconstructed from:

1. the notes preserved with the dataset;
2. BrainVision marker sequences and their latencies;
3. the `EEG.history` field in preserved MATLAB v7.3 EEGLAB files;
4. the event composition of the saved go, successful-stop, and failed-stop datasets.

## Original processing sequence

The preserved EEGLAB history records the following operations:

1. import the BrainVision `.vhdr` recording with all 129 channels;
2. save the continuous `.set` file;
3. band-pass filter from 1 to 40 Hz with `pop_eegfiltnew`;
4. attach `standard_BEM/elec/standard_1005.elc` channel locations;
5. visually inspect the continuous recording;
6. remove `ECG`;
7. apply average reference;
8. create go epochs around `S17` and `S18`, from −1.5 to 3.0 s;
9. create stop epochs around `S19`, from −2.0 to 2.0 s;
10. run extended Infomax ICA (`runica`, extended mode), requesting PCA rank 124;
11. inspect the first 35 component maps and component time-frequency activity;
12. save the condition datasets.

The preserved condition-specific files for one recording provide the strongest
direct check of the recovered operational mapping:

- `S6` = correct go response;
- `S4` = incorrect or too-slow go response;
- `S5` after `S19` = failed inhibition;
- `S19` without a following `S5` before the next trial supports an inferred
  successful-stop label.

The remaining recordings provide sequence-level consistency checks, not an
independent behavioral ground truth for this mapping.

Participant-level counts and marker-onset intervals are intentionally omitted
from this public repository. The intervals are not described as behavioral
reaction times because their acquisition meaning has not been independently
validated.

The current Python workflow is the executable reconstruction. The MATLAB script
is retained only as a readable historical reference and does not implement the
current ambiguity states, lineage, or provenance contracts.

## Reproducibility issues found

### EOG was not typed

After ECG removal, the preserved continuous `.set` contains 128 channels: 127 scalp EEG channels and one `EOG`. All channel `type` fields are empty, and EOG has no spatial coordinates. The recorded call `pop_reref(EEG, [])` contains no exclusion list. Therefore EOG may have been included in the original average reference and ICA.

The reconstructed pipeline assigns ECG/EOG channel types before rereferencing
and uses EEG channels only for the average reference. Any later ICA fit must
also be restricted to EEG channels and recorded separately.

### Bad-channel interpolation is not logged

The notes mention visual bad-channel detection and interpolation, but the saved `EEG.history` does not contain the removed-channel list or an interpolation command. Exact original bad-channel decisions cannot be reproduced.

The new pipeline reports automated candidates but requires visual confirmation and an explicit saved bad-channel list before spherical interpolation.

### ICA rejection is not explicit

The files preserve ICA weights (105 components for go; 104 for both stop datasets), but the saved reject flags contain no selected components. The history contains a component-inspection call followed by `pop_subcomp` without explicit component indices. Exact original component rejection cannot be recovered.

The new pipeline must save an explicit component table containing component number, decision, reason, and reviewer.

### Source-localization boundary

Only template electrode positions are available. There is no individual MRI or
digitized electrode geometry. Individual source localization is therefore out
of scope for this project.
