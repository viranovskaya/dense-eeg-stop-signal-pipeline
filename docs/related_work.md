# Related work and novelty boundary

This note defines the comparison set for the methods paper. It is not a claim
that the current workflow replaces any of these tools.

## Closest EEG preprocessing systems

| System | What it already contributes | Difference from this project |
| --- | --- | --- |
| PREP | Automated early-stage preprocessing, robust referencing, noisy-channel detection and per-recording reports. | The present work does not claim a new robust-reference method or a better automatic detector. It treats screening results as prompts and binds later human decisions to exact source and software identities. |
| HAPPE and HAPPE+ER | Standardized automated processing for short or high-artifact recordings, including quality metrics and ERP-oriented extensions. | The present work does not claim a broadly optimized cleaning recipe. It focuses on a fail-closed transition from review evidence to explicit channel, interval and component decisions. |
| Automagic | BIDS-compatible batch processing, quantitative quality metrics, stored settings and processing logs. | A processing log records what ran. The decision bundles here additionally constrain what downstream stages are allowed to consume and stop when the source, QC prompt package, decision bundle or runtime no longer matches. |
| EEG-IP-L / Lossless | Interactive quality control, signal-quality annotations, ICA decomposition and an explicit aim to preserve information. | Interactive review, annotation and the separation of review from later destructive processing are therefore not novelty claims. |
| PyLossless | A current MNE/BIDS implementation of the lossless approach, with non-destructive sensor, time and component flags, a browser review dashboard and analysis-specific rejection policies. | This is the closest contemporary comparison. The narrower contribution here is zero automatic cleaning or exclusion decisions plus immutable, exact-set-verified packages that fail closed when exact input, executable source, runtime, QC prompt, decision or ICA-solution identities drift. |
| MNE-BIDS-Pipeline | Configurable BIDS-native automation, MNE processing and structured reports. | This project is not a general alternative to MNE-BIDS-Pipeline. It concentrates on legacy event recovery and on provenance-bound manual gates that prevent unreviewed decisions from entering ICA or preprocessing. |
| EEGPrep | A fresh 2026 arXiv preprint implementing the default EEGLAB workflow in Python with stage-by-stage numerical comparison to MATLAB, BIDS input and BIDS derivatives. | The present work does not claim EEGLAB parity. Its validation target is decision and provenance integrity across two acquisition formats, plus deterministic execution under one pinned runtime. |
| CLEAN-EEG | Modular multi-site preprocessing with logging, plots and automated component classification. | The present work does not claim a new cross-site cleaning standard or automatic component classifier. Human component decisions remain required and solution-bound. |

## Data organization and computational provenance

EEG-BIDS standardizes raw EEG organization and metadata, including EEGLAB
`.set` and `.fdt` files. BIDS organization alone does not prove that a later
manual decision was made from the same source bytes, prompt package and
software context that a preprocessing run consumed.

DataLad and the FAIRly big framework provide a broader and more powerful model
for machine-actionable computational provenance and re-execution. This project
does not claim to replace them. Its contribution is domain-specific: the
meaning and completeness rules for channel, time-interval and ICA-component
decisions, followed by fail-closed checks at the EEG workflow boundaries.

The emerging BIDS provenance model includes file digests, generated-by
activities and explicit descriptions of manual steps. The current controlled
packages can inform a later BIDS-Derivatives export, but they are not yet
claimed to be a complete BIDS provenance graph or a BIDS-Derivatives dataset.

## Defensible novelty statement

The software contribution is a review-to-processing contract for dense EEG.
Screening creates evidence prompts but no cleaning decisions. Human channel,
interval and component decisions are stored separately, checked for complete
prompt coverage, and bound to the exact source, QC prompt package, captured
decision bytes, fitted ICA solution, executable source and runtime. Human-entered
evidence references document what was inspected; they are recorded text, not
cryptographic identities for the external viewer state. Each downstream stage
rechecks its upstream identities and publishes a new exact-set-verified output
only after all workflow gates pass.

This novelty statement is deliberately narrower than any of the following:

- a new artifact detector;
- a universally optimal preprocessing pipeline;
- the first interactive EEG quality-control system;
- the first BIDS EEG workflow;
- a replacement for general workflow provenance systems;
- proof that reviewed EEG is clean or scientifically valid.

## Evidence still needed for the paper

The implementation and public synthetic checks can support the software and
integrity claims. The real HBN path still needs completed channel and interval
review, completed ICA-component review and a controlled rerun. A second
independent review would permit agreement reporting, but it must remain
separate from the adjudicated decisions used for preprocessing.

## Core references

1. Bigdely-Shamlo N, Mullen T, Kothe C, Su KM, Robbins KA. The PREP pipeline: standardized preprocessing for large-scale EEG analysis. *Frontiers in Neuroinformatics*. 2015;9:16. https://doi.org/10.3389/fninf.2015.00016
2. Gabard-Durnam LJ, Mendez Leal AS, Wilkinson CL, Levin AR. The Harvard Automated Processing Pipeline for Electroencephalography (HAPPE). *Frontiers in Neuroscience*. 2018;12:97. https://doi.org/10.3389/fnins.2018.00097
3. Monachino AD, Lopez KL, Pierce LJ, Gabard-Durnam LJ. The HAPPE plus Event-Related (HAPPE+ER) software. *Developmental Cognitive Neuroscience*. 2022;57:101140. https://doi.org/10.1016/j.dcn.2022.101140
4. Pedroni A, Bahreini A, Langer N. Automagic: Standardized preprocessing of big EEG data. *NeuroImage*. 2019;200:460-473. https://doi.org/10.1016/j.neuroimage.2019.06.046
5. Pernet CR, Appelhoff S, Gorgolewski KJ, et al. EEG-BIDS, an extension to the brain imaging data structure for electroencephalography. *Scientific Data*. 2019;6:103. https://doi.org/10.1038/s41597-019-0104-8
6. Desjardins JA, van Noordt S, Huberty S, Segalowitz SJ, Elsabbagh M. EEG Integrated Platform Lossless preprocessing pipeline. *Journal of Neuroscience Methods*. 2021;347:108961. https://doi.org/10.1016/j.jneumeth.2020.108961
7. Halchenko YO, Meyer K, Poldrack B, et al. FAIRly big: A framework for computationally reproducible processing of large-scale data. *Scientific Data*. 2022;9:80. https://doi.org/10.1038/s41597-022-01163-2
8. Delorme A, Ranganath S, Kothe C, Jaiswal A, Makeig S, Aristimunha B. EEGPrep: a validated Python implementation of the EEGLAB preprocessing pipeline. arXiv:2607.16647. 2026. https://doi.org/10.48550/arXiv.2607.16647
9. Böttcher A, Wendiggensen P, Mückschel M, et al. Standardizing EEG preprocessing for cross-site integration: the CLEAN pipeline. *NeuroImage*. 2026;328:121812. https://doi.org/10.1016/j.neuroimage.2026.121812
10. Huberty S, Desjardins J, Collins T, Elsabbagh M, O'Reilly C. PyLossless: A non-destructive EEG processing pipeline. *Behavior Research Methods*. 2026;58:220. https://doi.org/10.3758/s13428-026-02997-z
11. Höchenberger R, Larson E, Gramfort A, et al. MNE-BIDS-Pipeline: v1.10.0. *Zenodo*. 2026. https://doi.org/10.5281/zenodo.19440738
12. Halchenko YO, Meyer K, Poldrack B, et al. DataLad: distributed system for joint management of code, data, and their relationship. *Journal of Open Source Software*. 2021;6(63):3262. https://doi.org/10.21105/joss.03262
