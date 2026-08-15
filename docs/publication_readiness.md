# Publication readiness

This checklist is the completion boundary for the Dense EEG methods/software
project. A passing software test suite is necessary but is not sufficient to
mark the project publication-ready.

| Requirement | Current evidence | Status |
| --- | --- | --- |
| Second public high-density EEG dataset | OpenNeuro `ds005508`, snapshot `1.0.1`; one 129-channel HBN contrast-change recording with a profile-bound BIDS/EEGLAB inventory | Complete |
| BIDS/EEGLAB adapter and task profile | Split-payload identity, inheritance-aware sidecars, channel units/types, event normalization, reference reconstruction and standard-template geometry are fail-closed and tested | Complete |
| Public-data QC | Full-recording temporal and line-noise screens produced an exact review package; screens are prompts, not decisions | Complete |
| Human channel and segment review | Controlled working copies contain completed review for all 31 channel prompts and 46 of 49 segment prompts; three intervals are packaged for independent review | **77 of 80 prompts reviewed; 3 pending** |
| Final channel/segment decision bundle | Finalizer and adversarial tests are complete; mutable working copies are not yet an immutable decision package | Waiting for the independent three-interval check and adjudication |
| Real HBN ICA review package | Consumer, diagnostics and immutable publication contract are complete | Waiting for finalized channel/segment decisions |
| Human ICA-component decisions | Solution-bound finalizer is complete | Waiting for the real ICA solution and investigator review |
| Real HBN preprocessing and epoch validation | Reviewed ICA application, post-ICA interpolation, epoch rejection and source-row lineage are implemented and validated on a deterministic public synthetic fixture | Waiting for component decisions |
| Deterministic aggregate results | Legacy aggregate execution, synthetic QC, synthetic ICA and synthetic BIDS/EEGLAB results are complete | Real HBN aggregate accounting pending |
| Independent reviews | Current public code, scientific claims and provenance/privacy snapshot passed independent review | Repeat after real HBN results and final manuscript changes |
| Paper package | Controlled manuscript draft, evidence matrix, public synthetic figure and tidy tables are present and hash-bound | Real HBN results, final privacy scan and investigator approval pending |

## Immediate controlled handoff

1. Send the controlled three-interval package only to an independent reviewer
   who has agreed to the non-redistribution boundary.
2. Collect decisions for the three remaining segment prompts without exposing
   the primary reviewer's choices.
3. Compare the two reviews, document any adjudication, and update the controlled
   working copy without changing the immutable QC prompts.
4. Run the channel/segment finalizer only after all 80 rows are complete. Do not
   fit ICA if any row is pending, incomplete, changed, or inconsistent with the
   immutable QC prompt package.

After the finalizer passes, the next controlled sequence is fixed:

1. create the real HBN ICA review package;
2. review every component and finalize the solution-bound ICA decision table;
3. run reviewed preprocessing and epoch accounting;
4. produce aggregate-only HBN results with no participant decision rows,
   intervals, paths or participant-linked hashes in the public paper package;
5. rerun the complete test suite, deterministic package checks, privacy scan and
   independent scientific, code and provenance reviews;
6. require explicit investigator approval before commit, push, release,
   submission or external transfer.
