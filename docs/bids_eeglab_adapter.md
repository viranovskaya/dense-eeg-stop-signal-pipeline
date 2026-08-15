# BIDS EEGLAB adapter

The adapter is a fail-closed inventory step for public BIDS EEG recordings stored as EEGLAB `.set` files. It does not reuse the recovered stop-signal event model.

The first profile targets the `contrastChangeDetection` task in HBN-EEG Release 4 (`ds005508`, snapshot `1.0.1`). The profile requires:

- exact dataset and snapshot identity;
- a complete adapter-derived BIDS sidecar inheritance chain;
- 129 EEG channels at 500 Hz with microvolt source units;
- the declared Cz online reference, represented by the expected flat Cz channel;
- finite embedded sensor coordinates, without yet asserting their coordinate frame or interpolation suitability;
- exact row, value, onset, and sample agreement between raw annotations and `events.tsv`;
- separate left-target and right-target epoch definitions;
- explicit exclusion of administrative or undocumented markers from analysis.

Passing the inventory permits controlled non-spatial channel and temporal QC. It does not authorize interpolation, rereferencing, preprocessing, scientific epoch analysis, or publication. The provisional target epochs cover the first 800 ms of a 1,600-ms gradual contrast change and are review aids, not a canonical ERP or spectral-analysis definition.

The HBN profile also compares the embedded 129-channel layout with MNE's `GSN-HydroCel-129` montage after a similarity transform. This validates the channel-name-to-template mapping used for review and, after a manual bad-channel decision, template-based interpolation. It does not provide individual digitization or an individual head fit.

Cz is present as the flat online-reference channel. The processing profile retains it as EEG and includes it in the later average-reference transform, while reviewed bad channels are excluded from the reference estimate. This reconstructs the saved online reference consistently rather than treating Cz as a measured zero-voltage signal.

The QC stage screens the complete recording in nominal non-overlapping 20-second windows, including the final partial window. Cz remains in the source recording but is excluded from scalp-channel scoring. Relative temporal scores use a 1–40 Hz fourth-order zero-phase Butterworth filter; 60 Hz concentration is measured separately from the raw PSD and shown in a ranked review figure. Every temporal prompt is written to a pending segment-review template, and the union of temporal and line-noise prompts is written to a separate channel-review template. A window prompt is not treated as the final artifact boundary. No channel is marked bad automatically.

Completed channel and segment tables are not used directly. A separate finalization command validates exact prompt coverage, reviewer/date/evidence fields, channel decisions (`keep` or `interpolate`), and refined segment exclusions (`ica`, `epochs`, or `both`). Interpolation and exclusion decisions require a written rationale. Reviewers may add provenance-bound manual channel or segment findings that the screen missed. A segment exclusion is a global time mask across all channels; the row's `channel` identifies the evidence origin, not a channel-limited exclusion. The command creates a new immutable decision bundle bound to the exact dataset inputs, QC provenance, runtime, and source controls.

For local review, a separate static worksheet can be built outside both immutable packages. It links to the verified figures, presents one prompt at a time, saves mutable progress in the browser and exports the exact channel and segment TSV schemas. It does not suggest or validate a scientific decision. Browser state and downloads remain unverified working copies until the finalizer succeeds.
The builder verifies both immutable package identities and every linked figure
before creating the page. The browser does not repeat those hashes while a
figure is open. Use a controlled browser profile and download directory, rebuild
after any package change, and clear saved browser state after finalization.

```bash
python scripts/build_bids_eeglab_review_worksheet.py \
  --review-pack /controlled/review-pack \
  --qc /controlled/qc-package \
  --channel-decisions /working/channel_decisions.tsv \
  --segment-decisions /working/segment_decisions.tsv \
  --output /working/review_worksheet.html \
  --session-id reviewer-a
```

```bash
python scripts/finalize_bids_eeglab_review.py \
  --dataset-root /data/ds005508 \
  --set /data/ds005508/sub-..._eeg.set \
  --qc /controlled/qc-package \
  --channel-decisions /controlled/completed-channel-review.tsv \
  --segment-decisions /controlled/completed-segment-review.tsv \
  --participant-id NDARMJ495DE0 \
  --output /controlled/final-decisions
```

The command fails if a prompt is missing, still pending, has changed prompt fields, lacks reviewer metadata or evidence, or defines an invalid refined interval. A refined exclusion linked to a QC prompt must overlap that prompt. Manual segment additions use a distinct `manual-*` identifier and leave the QC prompt timing fields empty; all manual additions use the `manual_addition` reason and a written rationale. The command never edits the dataset, QC package, or supplied tables.

The ICA consumer accepts only that verified decision bundle. It marks reviewed bad channels before rereferencing, expands reviewed ICA exclusions by half the 1–40 Hz zero-phase FIR support, filters while skipping the guarded intervals, and then applies an EEG average reference. The flat Cz online-reference channel remains in the transform and is reconstructed from the saved channel voltages; reviewed bad channels are excluded from the reference estimate. Interpolation is intentionally deferred until after a reviewed ICA solution is applied.

```bash
python scripts/run_bids_eeglab_ica.py \
  --dataset-root /data/ds005508 \
  --set /data/ds005508/sub-..._eeg.set \
  --profile config/hbn_contrast_change.json \
  --qc /controlled/qc-package \
  --review-bundle /controlled/final-decisions \
  --participant-id NDARMJ495DE0 \
  --output /controlled/ica-review
```

The result is an immutable, exact-set-verified ICA review package. It contains a fitted solution, component diagnostics, a time-course/spectrum/property figure for every component, overview topographies, and a blank decision table. Source-variance cues omit the same BAD samples excluded from ICA fitting and record their sample denominator. The selected HBN recording has no dedicated EOG or ECG channel, so those correlations are recorded as unavailable (`NaN`), not interpreted as evidence of a clean component. Correlation flags remain review cues and no component is removed automatically. The end-to-end path is regression-tested on a deterministic synthetic EEGLAB/BIDS fixture. Running it on the HBN recording still requires completed human channel and segment decisions; the existing QC prompts do not authorize a real ICA fit.

Completed component decisions are also finalized before application. The finalizer requires one solution-bound `keep` or `exclude` row per component with a reason, reviewer, date, and inspected evidence. Reviewed preprocessing applies only those exclusions, interpolates reviewed bad channels after ICA, restores the average reference, and exports the cleaned continuous recording plus one baseline-corrected epoch file and lineage table per declared condition. Epoch accounting keeps input, retained, dropped, and reason counts visible. FIR guard intervals remain BAD annotations because their samples were deliberately skipped by filtering; epochs overlapping them are therefore rejected rather than mixing filtered and unfiltered samples.

```bash
python scripts/finalize_bids_eeglab_ica.py \
  --dataset-root /data/ds005508 \
  --set /data/ds005508/sub-..._eeg.set \
  --qc /controlled/qc-package \
  --review-bundle /controlled/final-decisions \
  --ica-review /controlled/ica-review \
  --decisions /controlled/completed-ica-decisions.csv \
  --participant-id NDARMJ495DE0 \
  --output /controlled/final-ica-decisions

python scripts/run_bids_eeglab_preprocess.py \
  --dataset-root /data/ds005508 \
  --set /data/ds005508/sub-..._eeg.set \
  --profile config/hbn_contrast_change.json \
  --qc /controlled/qc-package \
  --review-bundle /controlled/final-decisions \
  --ica-review /controlled/ica-review \
  --ica-decisions /controlled/final-ica-decisions \
  --participant-id NDARMJ495DE0 \
  --output /controlled/processed
```

The synthetic integration fixture reproduces the complete ICA-decision and preprocessing packages byte for byte across repeated runs. It includes one reviewed interpolation and one reviewed epoch rejection, and its lineage table retains both the condition-local event index and the exact source BIDS row index. That establishes software-path determinism under one pinned runtime; it is not evidence that the HBN recording is clean, that any real component should be removed, or that the provisional task epochs answer a scientific question.

The screen is deliberately conservative. Spatial robust scores can miss common-mode contamination, a short transient can be diluted inside a 20-second window, and moderate attenuation or bridging can remain below the fixed review threshold. Montage-derived neighbours are visual aids because the coordinate frame has not been independently validated. Reviewers must inspect the trace, neighbours, persistence, and spectrum before recording any channel decision or refining a segment boundary and scope.
