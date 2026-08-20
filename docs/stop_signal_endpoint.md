# Fixed stop-signal ERP endpoint

This study uses one bounded neural endpoint. It is not an SSRT analysis and it
does not treat the inferred response-absent class as direct evidence of a
successful inhibitory mechanism.

## Frozen definition

- lock: observed `S19` stop-signal onset;
- conditions: operational response-absent (`stop_successful` in the executable
  codebook) versus observed response-present (`stop_failed`);
- epoch coverage required: the trial-specific 200 ms pre-go baseline and the
  stop-locked measurement window must both be present;
- baseline: -0.2 to 0 seconds relative to each trial's go-stimulus onset,
  implemented inside the stop-signal-locked epoch using the observed delay;
- ROI: FC1, FC2, and FCz, averaged without post-hoc channel selection;
- measurement: mean amplitude from 0.25 to 0.45 seconds;
- participant contrast: response-absent minus response-present;
- eligibility: at least eight retained epochs in each condition;
- group estimate: equal-weight mean across eligible participants;
- uncertainty: deterministic participant bootstrap 95% interval, 10,000 draws,
  seed 20260817;
- test: exact two-sided sign-flip test across participant contrasts, interpreted
  under the null assumption that participant contrasts are exchangeable with
  respect to sign (equivalently, a symmetric contrast distribution around
  zero).

The aggregate summary reports retained stop-epoch counts separately for all,
eligible and excluded recordings, together with counts for each frozen
exclusion reason. The participant table also records descriptive behavioral diagnostics from all
classified trials: the classified and unclassified stop-trial denominator,
response probability among classified stop trials, observed SSD by outcome, go
RT from classified go trials with an observed response, failed-stop RT, and the
expected failed-stop-RT versus go-RT ordering. Correct and
incorrect-or-slow go responses enter this diagnostic; unresolved or invalid
marker sequences do not. These are task-validity context, not a second
inferential endpoint and not an SSRT estimate. In particular, the ERP contrast remains potentially associated with
the expected SSD difference between response-absent and response-present stop
trials; the SSD diagnostic makes that limitation visible rather than treating
the contrast as a pure measure of inhibition.

The eight-epoch minimum is a pragmatic quality gate, not a physiological or
literature-derived threshold. It was frozen before the final reviewed
preprocessing run, after the preliminary trial inventory was available, and
will not be lowered in response to the observed endpoint. The participant table
keeps the retained counts and exclusion reason visible.

The endpoint consumes only provenance-verified participant preprocessing
packages. It checks the 0.2--30 Hz ERP filter and requires exact agreement
between retained stop epochs and `stop_epoch_lineage.csv` before calculating a
value. It also refuses an exploratory dataset: every participant package must
bind the completed fixed-study interval manifest and a reviewed ICA solution
with a complete solution-bound component decision table.

```bash
python scripts/run_stop_signal_endpoint.py \
  --dataset-output /controlled/path/dataset-preprocessing \
  --output /controlled/path/stop-signal-endpoint
```

The output is a private derived package. `participant_endpoint.csv` contains
pseudonymous participant-level values and must not be copied into a public
release. `endpoint_summary.json` contains aggregate inference and is also kept
outside this public software repository.

The endpoint output must be a new path outside the preprocessing dataset. The
command verifies the preprocessing dataset and every participant package before
calculation, immediately before atomic publication, and once again after
publication. Any identity change removes the published endpoint package and
fails the run.

## Scientific boundary

The 0.2 Hz high-pass has a quantitative signal-to-noise and distortion rationale
for P3 mean-amplitude scoring, while 30 Hz preserves the conventional ERP band.
The published recommendation was derived from ERP CORE P3b data using a filter
implementation that is not identical to this MNE zero-phase FIR pipeline. It is
therefore an a priori rationale, not a claim that 0.2--30 Hz is optimal or
validated for this stop-signal dataset. The 250--450 ms interval brackets the
frontocentral stop-signal P3 around its approximately 300 ms peak; it is a fixed
study window, not a window copied from an identical prior analysis and not a
post-hoc peak or onset estimate. The trial classes remain operational because
the response-absent class is inferred from the marker sequence.

- Zhang, Garrett, and Luck (2024), [recommended ERP filter settings](https://doi.org/10.1111/psyp.14530)
- Diesburg, Wessel, and Jones (2024), [frontocentral stop-signal ERP timing](https://doi.org/10.1523/JNEUROSCI.2016-23.2024)
- Verbruggen et al. (2019), [stop-signal consensus guide](https://pmc.ncbi.nlm.nih.gov/articles/PMC6533084/)
