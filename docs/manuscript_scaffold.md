# Manuscript scaffold

## Working title

Provenance-bound review and preprocessing for legacy and BIDS dense-array EEG

## Research question

Can a dense-array EEG workflow make event reconstruction, quality review, preprocessing decisions and output lineage explicit enough to be checked and repeated without turning heuristics into automatic scientific decisions?

## Main contribution

The contribution is the decision and evidence architecture, not a new artifact classifier. The software separates four stages:

1. source and metadata validation;
2. review prompts with no automatic channel or segment decisions;
3. immutable human decisions bound to the exact inputs and software context;
4. deterministic application, accounting and output verification.

The BrainVision path shows how this architecture recovers a legacy stop-signal workflow. The BIDS/EEGLAB path tests whether the same principles transfer to a second public 129-channel dataset with different events, metadata and reference semantics.

## Methods

### Evidence partitions

Report the five evidence partitions in `paper_validation_matrix.json` separately. Do not combine synthetic known truth, controlled private execution and public participant QC into one validation sample.

### Legacy BrainVision workflow

Describe the recovered marker state machine, explicit successful-stop inference and acquisition-integrity checks. State the processing order directly: continuous filtering, reviewed bad-channel marking, initial average reference, reviewed ICA when used, interpolation, final average reference and epoching. FIF and EEGLAB export are supported by the code and synthetic tests; EEGLAB export was disabled in the controlled 10-recording rerun. State that this rerun covered 121.7 minutes, 129 channels and 1000 Hz. Participant-level outputs and decisions remain private.

### Synthetic QC benchmark

Describe the fixed corruption families, seed-level holdout, channel-window evaluation units, raw-PSD line-noise metric and clean-signal reference. Report primary and stress-condition results separately. Report preservation by unaffected, oracle-interpolated and other corrupted channel-band values rather than relying on one overall median.

### Reviewed ICA validation

Describe extended Infomax, the MNE small-angle stopping rule, explicit component decisions and the auxiliary-channel, spectral, topographic and temporal cues used during review. State that no component was removed automatically. Report the full denominator for spectral preservation and the C3/C4 task-peak checks.

### BIDS/EEGLAB adapter

Describe exact OpenNeuro snapshot identity, inheritance-aware sidecar discovery, split EEGLAB payload hashing, units, events, flat Cz reference reconstruction and template-layout validation. Explain that full-recording QC creates prompts only. Human channel, segment and component decisions are finalized into separate exact-set-verified packages before they can be consumed.

Use the decision semantics in [`human_review_protocol.md`](human_review_protocol.md).
If two independent reviews are completed, keep their agreement analysis
separate from the adjudicated tables used for preprocessing and retain the full
decision denominators.

### Filtering, interpolation and epochs

Reviewed ICA intervals are expanded by half the support of the same zero-phase FIR used downstream. Filtering skips both the exact intervals and their guards. ICA is fit before interpolation. Reviewed component exclusions are then applied, reviewed bad channels are interpolated, and the average reference is restored. Epochs overlapping reviewed or guard annotations are rejected. The minimum event requirement is enforced again after rejection. Each lineage row retains the exact source `events.tsv` row, onset, sample, condition-local index and drop reason.

### Reproducibility and integrity

For each generated atomic output package, distinguish the controls and tables captured as immutable bytes from large EEG payloads protected by initial and final identity hashes. Describe the executable source manifest, runtime and dependency identity, exact recursive output hashes, symlink and path checks, final upstream rechecks and atomic publication to a new path. Distinguish deterministic execution under one pinned runtime from cross-platform numerical equivalence.

## Results that can be written now

### Controlled legacy rerun

All 10 controlled recordings completed and produced provenance-verified participant outputs and a dataset aggregate. This supports execution and accounting claims. It does not provide independent ground truth for reconstructed events or artifact decisions.

### Synthetic QC benchmark

Across three reserved seeds, primary temporal detection had precision, recall and F1 of 1.00. Each recording contained 13 positive pairs among 1,135 primary channel-window pairs per recording. Across all three recordings, this was 39 positive pairs among 3,405 primary pairs. One additional 100-microvolt, 0.5-second electrode-pop stress positive was diluted within its 20-second window and missed in each seed. The all-temporal result therefore covered 42 positives among 3,408 pairs, with recall 0.929 and F1 0.963; equivalently, 1,136 all-temporal pairs per recording. Across all 1,524 channel-band values, the unaffected subset contained 1,464 values with median, 95th-percentile and maximum absolute errors of 0.012, 0.092 and 0.141 dB. The oracle-interpolated subset contained 24 values with errors of 6.450, 12.751 and 16.798 dB. The other corrupted subset contained 36 values with errors of 0.058, 23.293 and 23.774 dB. The overall median, 95th-percentile and maximum errors were 0.013, 0.100 and 23.774 dB. The unaffected median absolute power error 0.012 dB and 95th-percentile error 0.092 dB did not characterize the two corrupted subsets.

### Synthetic ICA validation

On the fixed seed-4401 fixture, the MNE small-angle stopping rule was met after 530 iterations. Two reviewed components were excluded. Maximum absolute EEG-EOG correlation changed from 0.295 to 0.028 and maximum absolute EEG-ECG correlation from 0.176 to 0.049. Across 508 channel-band values, median, 95th-percentile and maximum absolute changes were 0.082, 0.645 and 0.994 dB. The largest C3/C4 peak-amplitude change was 0.230 microvolts and the largest latency change was 0 ms. These are integration checks on one synthetic fixture, not estimates of participant-level performance.

### BIDS/EEGLAB software path

The synthetic BIDS/EEGLAB fixture reproduced complete ICA-decision and preprocessing packages byte for byte under the pinned runtime. The test applies one component exclusion, one post-ICA channel interpolation and one reviewed epoch rejection while preserving the exact source BIDS-row lineage. The real HBN recording remains at inventory and controlled-QC status. No real HBN cleaning or task-effect result is reported.

## Results that remain blocked

- real HBN channel and segment decisions;
- real HBN ICA-component decisions;
- real HBN before/after signal-preservation summaries;
- any participant-level task-effect estimate;
- independent reviewer agreement on manual decisions;
- external generalization to another laboratory or acquisition system.

## Planned tables

1. Evidence partitions, public availability and allowed claims.
2. Provenance and fail-closed gates by workflow stage.
3. Synthetic QC detection and preservation metrics with explicit denominators.
4. Synthetic ICA before/after checks and decision accounting.
5. Aggregate controlled-run execution and provenance accounting, without participant rows or mappings.

## Planned figures

1. Workflow diagram from source validation to immutable decisions and verified outputs. A vector draft is available at [`figures/review_to_processing_contract.svg`](figures/review_to_processing_contract.svg).
2. Synthetic corruption examples and temporal detection summary.
3. ICA component-review evidence and bounded signal-preservation checks from the public synthetic seed-4401 fixture only.
4. BIDS/EEGLAB decision flow showing the human gates before real processing.

## Related-work and novelty boundary

The comparison set and core references are recorded in
[`related_work.md`](related_work.md). PyLossless is the closest contemporary
comparator because it already combines BIDS/MNE processing, non-destructive
sensor, time and component flags, interactive review and analysis-specific
rejection policies. EEG-IP-L/Lossless established the earlier lossless model.
PREP, HAPPE, Automagic, MNE-BIDS-Pipeline, the fresh EEGPrep arXiv preprint and
CLEAN-EEG also cover important parts of standardized or automated EEG
preprocessing. The paper must not claim the first interactive review system,
the first separation of review from destructive processing, the first BIDS EEG
pipeline, a new artifact classifier or a generally superior cleaning recipe.

The narrower contribution is zero automatic cleaning or exclusion decisions
plus immutable binding of complete human channel, interval and component decisions to exact source,
software, runtime, QC-prompt and ICA-solution identities. Downstream stages fail
closed when those identities no longer match, preserve source-event lineage and
produce only exact-set-verified outputs. General workflow provenance systems
such as DataLad remain broader than this implementation; the distinction is the
EEG-specific decision contract rather than a replacement provenance framework.

## Claims not to make

- automatic bad-channel, segment or ICA classification;
- proof that participant EEG is clean;
- validated ERP, time-frequency or stop-signal effects;
- universal thresholds or cross-dataset generalization;
- clinical suitability;
- privacy, legal or regulatory certification.

## Submission gate

Before submission, regenerate every machine result from an exact package, verify every number against its JSON source, review all public text and figures for participant identifiers or mappings, and obtain a separate scientific and provenance review of the final manuscript and supplement. Submission, public release and external transfer still require explicit final investigator approval.
