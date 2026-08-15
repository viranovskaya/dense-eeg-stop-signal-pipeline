# Controlled human-review protocol

This protocol defines the manual stage for the public HBN BIDS/EEGLAB example.
It does not turn the QC thresholds into automatic cleaning rules.

## Review materials

The reviewer uses one exact QC package and its bound review pack. The original
decision templates remain unchanged. Each reviewer makes separate working
copies outside the QC and review-pack directories.

Working copies are mutable, unverified inputs. Matching their initial hashes
to the templates proves only that the session started from the correct copies.
Edited decisions become provenance-bound only after the finalizer validates
them and successfully publishes an immutable decision bundle.

The optional local worksheet keeps the evidence package read-only, shows one
prompt at a time, stores progress in that browser and exports the same TSV
schemas. It is a convenience interface, not a decision engine or provenance
record. Its browser state and downloaded files remain mutable and unverified.
Use it only in a controlled browser profile, keep downloads inside the
controlled review workspace, and clear its saved state after the completed TSV
files have passed finalization. If browser storage is unavailable, keep the page
open and export both tables frequently. Linked figures are hash-verified when
the worksheet is built; the browser does not reverify them while they are open,
so rebuild the worksheet if either immutable package changes.
Give each independent reviewer a distinct worksheet `session-id` and controlled
browser profile. Reusing either can expose another reviewer's saved state and
break the independence assumption.

For every decision, record:

- reviewer identity;
- review date in `YYYY-MM-DD` format;
- the inspected figure, raw-viewer interval or other evidence;
- a short rationale whenever data are excluded or a channel is interpolated.

The static panels support triage. Final interval boundaries must be confirmed
in a zoomable raw-data viewer because a 20-second prompt is not an artifact
boundary.

## Channel decisions

Review every prompted channel using the full raw and filtered trace, the raw
PSD relative to the scalp median, the ranked line-noise plot, the temporal
screen and neighbouring channels.

`keep` means that the available evidence does not justify interpolation. It
does not assert that the channel is artifact-free.

`interpolate` is appropriate only when the problem is persistent or recurrent,
spatially channel-specific and not better explained by a common-mode event,
reference behavior or plausible physiology. The rationale must state the
evidence. Template-based interpolation is permitted only because the separate
geometry gate passed for this recording; it does not imply an individual head
fit.

If the reviewer identifies an unprompted channel problem, add a row with the
real channel name, `candidate_reason=manual_addition` and a written rationale.

## Segment decisions

Review every `candidate_id` using its raw and filtered target trace, montage
neighbours, context outside the red prompt boundaries and the zoomable raw
recording.

`keep` means that the prompt does not justify a time exclusion. Leave refined
timing and scope empty.

`exclude` requires a refined onset, positive duration and one scope:

- `ica`: omit the interval from ICA fitting but do not automatically reject an
  otherwise valid task epoch;
- `epochs`: retain the interval for ICA fitting but reject task epochs that
  overlap it;
- `both`: omit it from ICA fitting and reject overlapping task epochs.

Every exclusion is a global all-channel time mask. The `channel` field records
where the evidence was observed; it does not make the exclusion channel-local.
For `epochs` and `both`, an epoch is rejected when it overlaps the refined
interval or its provenance-recorded FIR guard. This cost must be considered
before excluding a channel-local transient.

Use `both` when the same corruption is unsuitable for decomposition and for
task-level data. Use a single-target scope only when the rationale explains why
the other target remains scientifically acceptable. A refined interval linked
to a QC prompt must overlap that prompt. The downstream FIR guard is computed
from the reviewed interval; the reviewer should not manually add filter padding.

For an unprompted interval, use a unique `manual-*` candidate ID,
`candidate_reason=manual_addition`, the evidence channel or `all`, and a written
rationale.

## ICA-component review

ICA starts only after channel and segment decisions have been finalized into a
verified bundle. Review every component using its topography, time course,
spectrum, source-variance denominator and available auxiliary-channel cues.
The selected HBN recording has no dedicated EOG or ECG channel, so missing
correlations are not evidence that a component is neural.

`keep` is the conservative decision when artifact evidence is ambiguous.
`exclude` requires a component-specific rationale and inspected evidence. No
component is removed from a threshold or label alone.

## Independent review and adjudication

If two reviewers are available, they should complete separate copies without
seeing each other's decisions. Preserve both original tables. Report channel
and segment decision agreement over the complete original prompt denominator,
with the two-by-two decision counts. Report manual additions separately because
they do not share a natural prompt identifier. Match manual channel additions
by exact channel name. Match manual intervals with a deterministic one-to-one
assignment that maximizes temporal intersection over union (IoU), accepting
only pairs with IoU at least 0.5. Break equal-IoU ties by the smaller absolute
onset difference and then original row order. Do not require the evidence
channel to match because two reviewers may notice the same global artifact in
different traces; report evidence-channel agreement separately. Report
unmatched additions rather than forcing a match. Cohen's kappa may be added
when both decision classes occur, but raw agreement must remain visible because
rare exclusions can make kappa unstable.

For prompted intervals excluded by both reviewers, report signed and absolute
onset and offset differences in seconds, including medians and ranges, separately
from categorical agreement. Scope agreement is reported only among prompts both
reviewers excluded.

ICA-component agreement uses the complete component denominator. Report the
`keep`/`exclude` two-by-two counts and raw agreement; add Cohen's kappa only when
both decision classes occur. Preserve both independent component tables before
adjudication. Resolve disagreements in a third adjudicated copy. Only the
adjudicated tables may be finalized for downstream processing; agreement
statistics must be computed from the untouched independent copies.

This agreement describes decisions conditional on the same QC prompts and
candidate-reason definitions. It does not estimate agreement for artifacts that
the screening stage failed to prompt; manual additions and unmatched findings
must remain visible as separate completeness evidence.

## Completion gate

The channel and segment finalizer must pass without pending rows, missing
evidence, altered prompt fields or invalid timing. Passing the finalizer
authorizes creation of an ICA review package only. It does not authorize ICA
component exclusion, preprocessing, task-effect analysis or publication.
