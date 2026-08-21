# Fixed-study interval decisions

This table records only temporal exclusions reviewed for the ten stop-signal
recordings. It is not a general artifact schema. A QC flag is not an exclusion,
and a local transient is not converted into whole-channel interpolation.

## Required columns

`participant_id, interval_id, decision, start_s, stop_s, scope, reason,
reviewer, reviewed_at, evidence`

All ten input participants must be represented exactly. An excluded interval uses
`decision=exclude`, finite recording-relative bounds, and one of these scopes:

- `ica`: omit the interval from ICA fitting only;
- `epochs`: reject every task epoch that overlaps the interval only;
- `both`: apply both rules.

If review finds no temporal exclusion, the participant receives one
`decision=none` row with `interval_id=none`, blank bounds and blank scope. The
reason, reviewer, ISO review date and evidence reference remain required. A
`none` row cannot be mixed with exclusions. Duplicate IDs, overlapping
reviewed intervals, missing participants, unknown participants, invalid bounds
and changed manifest bytes stop the run.

## Filter support

The raw reviewed bounds identify the visible artifact. A zero-phase FIR filter
can spread its contribution beyond those bounds. The pipeline therefore adds
half of the exact FIR length on each side before ICA fitting or epoch rejection.
The guard is calculated from the sampling rate and the relevant filter, not
entered by hand. With the current MNE FIR settings at 1000 Hz it is 1.65 s for
the 1--40 Hz ICA stream and 8.25 s for the 0.2--30 Hz ERP stream.

The provenance record retains the original interval, guarded interval, scope,
reason, manifest hash and participant-decision hash. Every epoch touching a
guarded `epochs` or `both` interval is dropped, including overlap in the
baseline or measurement window. Its `BAD_fixed_epochs_*` reason remains in the
epoch lineage table.

## Scientific boundary

Reviewers set the interval bounds and scope from the raw evidence. The code
validates and applies those decisions but does not infer missing boundaries.
ICA review and final preprocessing must use the same manifest identity. A new
or edited manifest requires new ICA review packages and a new preprocessing
output.
