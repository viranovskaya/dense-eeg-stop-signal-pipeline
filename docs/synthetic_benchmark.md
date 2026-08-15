# Synthetic corruption benchmark

## Question

The benchmark asks two separate questions:

1. Does the current QC screen point a reviewer to the channel and 20-second
   window that contains a known corruption?
2. How much does a later cleaning decision change signal that was not
   corrupted?

Finding a review candidate is not the same as cleaning data well. Detection and
signal preservation are therefore reported separately.

## Public synthetic recording

The full profile generates 127 EEG channels on the standard 10-05 montage plus
one EOG and one ECG channel. It contains reproducible coloured background EEG,
alpha and beta rhythms, simple ocular and cardiac source projections,
stop-signal markers, and a small known event-locked C3/C4 signal. No participant
data are used. The generator is a software fixture, not a realistic model of
spatial EEG covariance.

Five channel-local corruption families are currently defined:

- persistent broadband noise;
- an intermittent oscillatory burst;
- a short electrode pop;
- a partially flat channel;
- 50 Hz line noise.

Each injected interval is stored as ground truth with its seed, channel, sample
bounds, duration and amplitude. Calibration, seed-level holdout and stress seeds
are fixed before thresholds are assessed. The holdout changes random seeds only:
channels, intervals, amplitudes and corruption families remain fixed. It does
not test new artifact conditions or external generalization.

The 100-µV, 0.5-second electrode pop is an explicit stress case. Its purpose is
to measure dilution of a sub-second event inside a 20-second window, not to
redefine the threshold until the event is found. Primary, stress and all-case
results remain separate in the output.

## Detection measures

The primary unit is a channel-window pair, matching the current non-overlapping
QC scan. The benchmark reports true positives, false positives, false negatives,
true negatives, precision, recall and F1. Results are also retained by
corruption family and seed so that a good average cannot hide a failed case.

The temporal score covers amplitude and flat-segment review prompts. Persistent
50 Hz contamination is assessed separately with the existing raw-PSD line-noise
ratio, at channel level. It is not counted as a failure of the 1--40 Hz temporal
amplitude screen, which is not designed to detect it.

The current 20-second scan can dilute short events and its results depend on
window alignment. Those are properties to measure, not benchmark details to
remove after seeing the results.

## Preservation measures

The clean generator output is kept as a known reference. Correlation, RMSE and
normalized RMSE are calculated separately for known-clean and known-corrupted
samples. A predeclared oracle decision table marks only persistent broadband
noise and partial flatness for whole-channel interpolation. It is used to test
the signal transformation, not to imitate automatic channel rejection. The
same filter and reference are applied to the clean reference, and C3/C4 event
amplitude and latency errors are reported after processing. Delta, theta, alpha
and beta power errors are retained by channel and summarized with the median,
95th percentile and maximum for unaffected, oracle-interpolated and other
corrupted channels. The marker stream contains
complete go, successful-stop and failed-stop sequences, and every trial start
must remain accounted for after reconstruction.

## Scope

This is a controlled software benchmark. It tests whether the implementation
behaves as intended under declared synthetic conditions. The oracle
interpolation is a code-path contract, not evidence that interpolation can
recover unknown biological signals. The benchmark does not estimate artifact
prevalence in real recordings, validate universal QC thresholds, or replace
review of participant data.

Run the held-out seeds with:

```bash
python scripts/run_synthetic_benchmark.py \
  --seed-group held_out \
  --export-brainvision \
  --output results/synthetic-held-out
```

The output contains corruption truth, detection comparisons, preservation
metrics, aggregate summaries, a source-bound provenance record and one
BrainVision round-trip example. The provenance record binds the exact output
set, sizes, SHA-256 hashes, Python and package versions, and dependency-file
hashes. It is verified before the atomic publish. The output path must be new.

The tracked [vector validation figure](paper_assets/figures/synthetic_qc_validation.svg) and
[tidy aggregate table](paper_assets/tables/synthetic_qc_validation.csv) are generated from
the held-out summary with `scripts/build_paper_assets.py`. The figure keeps the
stress-inclusive miss and all preservation subsets visible.
