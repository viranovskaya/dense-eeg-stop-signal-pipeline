# ICA component decisions

## Why this is separate

The historical EEGLAB files contain ICA weights but do not preserve a reliable
list of rejected components. This workflow does not guess those choices. It
fits a new solution and records every new decision.

## Processing order

1. Load the BrainVision recording and assign EEG, EOG and ECG channel types.
2. Apply the declared 1--40 Hz filter to EEG and EOG.
3. Mark manually confirmed bad EEG channels so they are excluded from the
   reference estimate and ICA fit.
4. Apply an EEG-only average reference.
5. Estimate the rank of the reviewed good EEG channels and fit extended
   Infomax with a fixed random seed and an explicit small-angle stopping rule.
   A fit that reaches the declared iteration limit stops the workflow instead
   of becoming a review candidate.
6. Save the ICA solution, component diagnostics, topography overview and
   detailed plots for correlation-based review candidates.
7. Copy the decision template outside the provenance-verified review package
   and record one explicit `keep` or `exclude` decision for every component.
8. Verify that the completed table matches the exact ICA solution SHA-256.
9. Apply only components marked `exclude`.
10. Remove ECG from the analysis output, interpolate reviewed bad EEG channels
    and restore the final EEG-only average reference.

The EOG and ECG correlations are prompts for inspection. The code never turns
them into automatic component rejection. No single summary measure is treated
as an artifact label. Review combines the scalp projection, activation time
course, spectrum and available auxiliary-channel evidence. In particular,
elevated 20--40 Hz power alone does not establish a muscle component; the
interpretation also requires compatible peripheral or focal, spatially
non-smooth and temporal evidence. If auxiliary evidence is unavailable, a
component is not assigned an ocular or cardiac label from its map alone.
Unresolved components are retained in the primary analysis and may be removed
only in a clearly labelled sensitivity branch.

ICA fitting rejects segments already marked with MNE `BAD_*` annotations. The
temporal channel-QC scan does not itself create segment annotations. The fixed
study therefore requires a separate, complete provenance-bound interval table;
without it the ICA and preprocessing commands stop rather than silently fit on
unreviewed transient candidates.

## Decision table

Each row contains:

- participant code;
- ICA solution SHA-256;
- zero-based component number;
- `keep` or `exclude`;
- a short reason;
- reviewer and review date;
- the evidence used for the decision.

Every component must appear exactly once. This includes components that are
kept. A blank row, duplicate component, unknown decision or changed solution
stops the run.

## Evidence boundary

The review package is private derived data. Its topographies, component traces,
spectra and auxiliary-channel relationships may still disclose information
about a recording and must not be committed or shared without review.

This workflow makes current component decisions auditable. It does not recover
the undocumented historical ICA exclusions, validate a universal correlation
threshold or prove that removed activity is non-neural.
