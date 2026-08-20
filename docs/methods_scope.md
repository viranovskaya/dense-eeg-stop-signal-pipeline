# Methods scope

## Main question

Can incomplete legacy dense-EEG stop-signal records be reconstructed into an
auditable trial-level dataset in which every event classification and
preprocessing decision is traceable to its source evidence?

## Evidence used for the event model

The current event definitions come from preserved notes, BrainVision marker
sequences, EEGLAB history, and the event composition of saved condition files.
The saved condition files for one preserved recording provide the strongest
direct check of the go, successful-stop, and failed-stop mapping. The remaining
recordings support sequence-level checks, but they are not an independent
ground truth.

## Rules for reconstruction

- The codebook is the executable source of marker meanings.
- A directly classified trial must contain the marker required by its rule.
- An absent response marker supports an inferred successful-stop label only
  when the next trial start closes the search interval. It is never presented
  as a directly observed outcome.
- Unknown, conflicting, and truncated sequences remain visible in the output.
- Every detected trial start must appear in the reconciliation table.
- Marker-onset intervals are acquisition integrity checks, not validated
  behavioral reaction times. Broad limits in `config/analysis.json` catch
  implausibly distant markers. They were checked for the controlled rerun and
  must be reassessed before applying the workflow to a new dataset.
- `low`, `medium`, and `high` describe confidence in a reconstruction rule.
  They are qualitative evidence labels, not probabilities or model scores.
- Exploratory stop-signal-locked C3/C4 QC plots are shown without baseline
  correction because the pre-stop interval can contain the go stimulus. The
  fixed FC1/FC2/FCz endpoint instead uses each trial's 200 ms pre-go baseline,
  located from the observed stimulus-to-stop-signal delay.
- Five representative windows are used only for compact overview figures.
  Channel review candidates also come from a non-overlapping 20-second scan
  covering every recorded sample, including the final partial window.
- The temporal scan uses within-window across-channel robust z scores to find
  local spatial outliers. A flag opens manual review; it does not interpolate
  a channel or reject an epoch automatically.
- ICA is fitted only to reviewed good EEG channels after the 1--40 Hz filter
  and EEG-only average reference. EOG and ECG remain available as diagnostic
  targets during review.
- The reviewed ICA solution is applied to the 0.2--30 Hz ERP stream before
  bad-channel interpolation and the final average reference.
- Component correlations and single summary measures are screening cues, not
  artifact labels. Every component must receive an explicit `keep` or
  `exclude` decision based on convergent evidence from its topography,
  activation time course, spectrum and, when available, auxiliary-channel
  relationship. A raised 20--40 Hz fraction alone is insufficient to label a
  component as muscle: that interpretation also requires a compatible
  peripheral or focal, spatially non-smooth projection and temporal behavior.
  In the absence of EOG or ECG evidence, a component is not labelled ocular or
  cardiac from its map alone. An unresolved component is retained in the
  primary branch and may be excluded only in a clearly labelled sensitivity
  branch.
- A manual re-review performed after downstream results have been inspected may
  confirm a frozen decision. If it changes a decision, the changed version is
  post hoc and must be reported as a sensitivity analysis; it cannot silently
  replace the frozen primary preprocessing branch.
- Existing `BAD_*` annotations are excluded from the ICA fit. Temporal QC
  windows are not converted into segment exclusions automatically. The fixed
  study requires an explicit, provenance-bound interval table. A local
  transient can be omitted from ICA fitting, task epochs, or both without
  turning its channel into a persistent interpolation decision.
- Reviewed interval bounds are expanded by half the exact zero-phase FIR
  support for the relevant stream. Every epoch overlapping a guarded exclusion
  is dropped, including overlap in its baseline or measurement window. The
  original and applied bounds remain separate in provenance and lineage.
- A participant with no temporal exclusions still has one explicit completed
  `none` row. Missing reviews, unknown participants, overlaps or changed table
  bytes stop the fixed-study run.
- The decision table is bound to the exact ICA solution by SHA-256. A partial
  table, a changed solution or an unreviewed component stops preprocessing.

## What this project can show

- how the legacy event logic was recovered;
- whether the recovered rules account for the available marker sequences;
- which trials remain ambiguous or incomplete;
- which new preprocessing decisions were made and why;
- whether repeated runs reproduce the same trial and output accounting.

## What this project does not show

- that the reconstructed labels are an independent behavioral ground truth;
- that the workflow generalizes to every stop-signal or dense-EEG dataset;
- that preprocessing proves a neural mechanism of inhibition;
- that SSRT is valid without the recommended behavioral checks;
- that template source localization is an individual anatomical result.
