# Methods

This is the technical companion to the [visual walkthrough](https://html-preview.github.io/?url=https://github.com/davisbone/Harmonic-EEG/blob/main/docs/explainer.html) and [MS2 notebook](../ms2_data.ipynb). It explains the measurements, the choices behind the pipeline, the exact processing recipe, and the limits of the resulting table. The [README](../README.md) is the shortest route to running it.

## Concepts and the data path

`Published EEG + event markers → verified recordings → blockwise cleaning → eligible event clips → 21 voltage summaries per retained event`

| Term | Meaning here |
| --- | --- |
| Channel | One electrode's voltage over time, measured relative to a reference. It is not an isolated brain region. |
| Block | One uninterrupted recording segment. The deposited session joins eight blocks; filtering must respect those joins. |
| Event / trial | A presentation of the three-chord progression. The middle chord defines time zero and the target label. |
| Epoch | The short signal clip around that event, nominally `[−100, 400)` ms. |
| Baseline | The average voltage just before the middle chord, subtracted separately from each channel. |
| Artifact | A contribution such as eye movement, muscle activity or electrode noise that can obscure the signal of interest. |
| ERP feature | A signed, event-aligned voltage average. A single-trial feature includes noise; it is not a pure evoked brain response. |
| QC | Quality control: the stated screening rules, not a guarantee of artifact-free EEG. |

The observations are repeated within participants. The final table has many event rows but only 27 retained participants; 32,935 rows do not provide 32,935 independent people for evaluating generalization.

## Question and observation unit

Can 21 early EEG voltage features tell standard from deviant middle chords in participants the model was not trained on? A row represents one participant’s presentation of a three-chord progression, aligned to chord two. Exemplar deviants change the middle chord’s inversion within a harmonic function. Function deviants change its harmonic function. Each chord lasts about 400 ms. Chord one starts at −400 ms, chord two at 0 ms, and chord three at +400 ms. Participants were instructed to press a button when they heard a deviant. [NEMAR dataset](https://nemar.org/dataset/on003570) · [Original EEG study](https://doi.org/10.1016/j.neuroimage.2021.118458) · [Original behavioral study](https://doi.org/10.1177/0305735618779444)

A **standard is defined by the current block's repeated progression**, not by a chord being universally “correct.” The experiment uses ii–IV–I, ii–IV6–I, ii–V–I and ii–V6–I progression types. An inversion changes which chord tone is lowest while retaining harmonic function; a function change alters the middle chord's role in the progression. A progression can therefore be standard in one block type and deviant in another. These musical categories come from the experimental markers; the pipeline does not infer them from EEG.

The outcome is the presented stimulus condition. The table supports offline condition classification and descriptive analysis. Chord acoustics, sequence context, motor preparation, and responses can contribute to EEG differences. Recorded response-related markers can occur before +400 ms. Zero-phase filtering also combines samples on both sides of a time point. The early windows therefore cannot establish a pre-response or causal neural signature of surprise.

## Source verification and event reconstruction

- The pipeline reads the [pinned NEMAR v1.0.0 release](https://data.nemar.org/on003570/v1.0.0/) using the [checked-in manifest](../eeg_pipeline/manifest-v1.0.0.json). It verifies file size and the release’s MD5 or Git blob hash. Missing files, changed bytes, or unexpected recording geometry stop the rebuild.
- The release audit covers 40 participants. Of these, 28 have complete EEG, `sub-001` and `sub-030` have verified prefixes with about 92% of the expected samples, and ten have less than half the expected EEG and are excluded at source. A matching checksum shows which published file was read; it does not mean the recording is complete. The audit records expected and available frames, any trailing bytes, and the SET header status.
- The 64-channel recording is sampled at 2048 Hz. An FDT frame is 64 little-endian float32 values in the SET/TSV channel order. Read only complete frames and ignore trailing partial-frame bytes. Convert the stored microvolts to volts for MNE and apply its BioSemi64 montage. Features and displayed amplitudes convert back to microvolts.
- Event order and onsets come from `events.tsv`. For readable SET files, the pipeline verifies the embedded event labels and timestamps against the TSV after converting one-based EEGLAB latencies. It checks three-chord order, adjacent chord spacing of about 400 ms, and 180 trials in each of eight blocks. A boundary marker falls between samples. The first sample after it starts the new block. Blocks end at the next splice or the last available frame boundary.
- A `two` → middle chord → `one` sequence yields one candidate. The middle marker starts with `fourRoot`, `four6`, `fiveRoot`, or `five6`. Suffixes `S` and `Sincorrect` map to `standard` (target 0). `deviantF` and `deviantFcorrect_F` map to `function_dev` (target 1). `deviantE` and `deviantEcorrect_E` map to `exemplar_dev` (target 1). The suffixes encode behavioral information as well, but only the stimulus mapping enters this table. Each candidate has an identity even if its EEG is unavailable.

## Why this representation and these processing choices?

| Choice | Purpose and tradeoff |
| --- | --- |
| 256 Hz rather than 2048 Hz | Reduces computation eightfold after antialias filtering. Its 128 Hz Nyquist limit accommodates both the 40 Hz analysis band and the separate 100 Hz ICA fitting band. |
| Separate analysis and ICA fitting copies | The final voltage summaries use 0.1–40 Hz. ICA/ICLabel uses 1–100 Hz data appropriate to its fitting and classification procedure. The fitted spatial correction is applied back to the analysis signal; ICA components are not the 21 predictors. |
| Interpolation plus average reference | Estimate an unusable electrode from the others, then subtract the same-time average across all channels. Interpolated channels are estimates, not recovered measurements. A noisy or repaired channel can affect other channels through the reference, which is why QC checks all 64 even though the regions use 38 distinct electrodes. |
| Seven fixed regions and three early windows | Produces a small, interpretable feature set instead of hundreds of channel/time/frequency inputs. Averaging trades spatial and temporal detail for simpler summaries; it does not guarantee noise cancellation. The windows are not claimed to isolate named ERP components. |
| Fixed screening thresholds | Makes inclusion decisions inspectable and consistent. The six-channel cap and 60% retention cutoff are practical choices, not uniquely correct biological boundaries. No cutoff was adjusted to obtain the current cohort size or maximize a classifier's score. |
| Zero-phase filtering | Avoids a systematic phase delay, but uses both earlier and later samples. This suits offline analysis and limits any claim that the features contain only activity occurring before 400 ms. |

The 60/120 Hz notch is applied to the broadband signal used for channel diagnostics and ICA before the analysis low-pass. It is not an extra source of predictive features.

## Cleaning and QC

The pipeline processes participants sequentially. Its signal and channel decisions use EEG, block boundaries, and middle-chord onset times to locate epochs. Stimulus labels and response timing do not enter those decisions. The thresholds below are practical cutoffs that are the same for every participant. The cohort size follows from them; it was not a target. [Configuration](../eeg_pipeline/config.py) · [Processing code](../eeg_pipeline/processing.py)

1. **Downsample and filter within each block.** Resample 2048 → 256 Hz by an exact factor of eight with reflected padding and a 161-tap symmetric Kaiser polyphase FIR (β = 5). Apply a 0.1 Hz high-pass and 60/120 Hz notch using zero-phase FIRs. The separate 40 Hz low-pass follows ICA application and completes the analysis band.
2. **Find persistent bad channels.** Across the high-passed, notch-filtered blocks, flag channels with nonfinite samples, zero median absolute deviation (MAD), or MAD below 0.1 times the median finite channel MAD. On the remaining channels, run PyPREP on average-referenced block interiors with the broadband FIR edge support removed. Run robust amplitude deviation (`|z| > 5`), high-frequency noise (`z > 5`), and correlation/dropout (more than 20% of one-second windows, with correlation threshold 0.4). Use the blockwise flags because PyPREP filters internally. A channel is persistently bad if flagged in blocks that together cover at least 50% of the tested duration. The participant audit retains block flags, tested sample counts, and flagged-duration fractions. This duration rule applies to PyPREP flags. The MAD screen uses the whole available recording. [PyPREP NoisyChannels](https://pyprep.readthedocs.io/en/stable/generated/pyprep.NoisyChannels.html)
3. **Repair and reference.** Use spherical-spline interpolation for the initial bad channels and compute a common average reference. An initial count above six ends processing for that participant.
4. **Fit and apply ICA.** Filter a separate copy to 1–100 Hz and use contiguous five-second segments within the corresponding block guard so ICLabel’s temporal inputs stay within blocks. Fit extended Picard with `ortho=False`, `extended=True`, seed 97, at most 500 iterations, and `min(30, estimated EEG rank)` components. Fit at 256 Hz (`decim=1`). Record the top ICLabel class and unrounded probability for each component. Remove components labeled eye blink, muscle artifact, heart beat, line noise, or channel noise when their top probability is strictly greater than 0.8. Apply the decomposition to the high-pass/notch analysis signal and then low-pass at 40 Hz. Nonconvergence interrupts the build. ICLabel probabilities are imperfect artifact evidence. [MNE ICA](https://mne.tools/stable/generated/mne.preprocessing.ICA.html) · [ICLabel API](https://mne.tools/mne-icalabel/stable/generated/api/mne_icalabel.iclabel.iclabel_label_components.html)
5. **Locate eligible epochs.** The intended half-open interval is `[−100, 400)` ms relative to chord two. It must fit within the recorded source, and both the intended and sample-aligned intervals must clear the block edges by the analysis guard. Align to the closest 256 Hz sample relative to the block start. The onset error is at most half a sample (about 1.95 ms). The guard sums `(taps − 1) / (2 × sample rate)` for the resampling, high-pass, notch, and analysis low-pass FIRs, then adds 1 s. The configured analysis guard is about 21.004 s. ICA fitting uses the fit-band FIR in place of the analysis low-pass for its own guard. Application of the fitted decomposition is spatial and adds no temporal support. With the pinned environment, the high-pass, notch and analysis low-pass have 8,449, 1,691 and 85 taps at 256 Hz; the resampling filter has 161 taps at 2048 Hz. Their combined half-support is about 20.004 s, followed by the separate 1 s safety margin. ICA fitting uses a 22.488 s guard. This is finite filter support, not an assertion that all samples in that support contribute equally.
6. **Make one intermittent-channel repair pass.** For each eligible epoch with a largest finite peak-to-peak amplitude above 150 µV, count its highest-amplitude channel. A previously unrepaired channel qualifies if it is the worst offender in more than 5% of all eligible epochs. If the union of initial and intermittent channels contains at most six channels, interpolate that union from the other channels and average-reference again. This is a single repair pass. The final epochs are average-referenced even when no second repair is needed. Flat or nonfinite epochs are evaluated by the final screen.
7. **Screen epochs and participants.** For each eligible epoch, check all 64 channels across the whole `[−100, 400)` ms interval. Fail the epoch if any sample is nonfinite, any channel’s peak-to-peak amplitude exceeds 150 µV, or any channel’s peak-to-peak amplitude is below 1 µV. Fail the participant if more than six channels require repair or if fewer than 60% of source- and boundary-eligible epochs pass signal QC. There is no per-condition minimum. Source-unavailable and edge-excluded epochs are outside the retention denominator.

## Feature construction

Subtract each channel’s mean over `[−100, 0)` ms from that channel’s retained epoch. This baseline falls during chord one. Average the baseline-corrected voltages equally across channels in each scalp region, then average the resulting trace in each of three half-open windows. The windows are `[80, 150)`, `[150, 250)`, and `[250, 400)` ms. At 256 Hz, the epoch contains 128 samples, the baseline contains 25, and the three windows contain 18, 25, and 39 samples respectively.

| Region | Channels |
| --- | --- |
| `Lfront` | AF3, F3, F5, F7, FC3, FC5 |
| `Rfront` | AF4, F4, F6, F8, FC4, FC6 |
| `FCmid` | Fz, F1, F2, FC1, FCz, FC2 |
| `Cmid` | C1, Cz, C2, CP1, CPz, CP2 |
| `Pmid` | P1, Pz, P2, PO3, POz, PO4 |
| `Ltemp` | T7, TP7, P7, P9 |
| `Rtemp` | T8, TP8, P8, P10 |

The 21 columns follow `erp_<region>_<window>_uv`, with window tokens `080_150`, `150_250`, and `250_400`. For example, `erp_Cmid_150_250_uv` is the signed mean voltage in microvolts across Cmid channels and the aligned `[150, 250)` ms samples, after channel-wise baseline subtraction. `PREDICTORS` in [the configuration](../eeg_pipeline/config.py) is the ordered allowlist. These are scalp summaries, with no source localization or spectral features.

A concrete example connects the formula to the table. For `sub-002_trial-0010`, average the six `FCmid` electrodes at each sample, subtract their pre-event baseline, then average each window. The stored values are **+5.878, +5.316 and +7.940 µV** for the three windows. Positive and negative signs are relative to the reference and baseline; they are not direct measures of “more” or “less” overall brain activity. These example values are checked against the Parquet table when the walkthrough is regenerated.

## Released cohort and exclusion accounting

These are the counts in the bundled version. Recompute them from the ledger and participant audit after any new build.

| Stage | Events remaining | What was removed at this stage |
| --- | ---: | --- |
| Candidate middle chords | 57,600 | 1,440 events for each of 40 deposited participants |
| Required source signal exists | 42,978 | 14,400 from ten severely truncated recordings, plus 222 beyond the two usable prefixes |
| Whole epoch clears block/filter edges | 37,477 | 5,501 boundary/guard exclusions |
| Participant passes initial channel-count screen | 33,705 | 3,772 eligible events from `sub-006`, `sub-023`, `sub-035` (7, 9, 13 bad channels) |
| Epoch passes signal checks | 32,935 | 770 amplitude failures; none were flatness or nonfinite failures in this release |
| Participant passes retention screen | 32,935 | No further losses; retained participants range from about 83.4% to 99.9% signal retention |

The final 27 participants comprise 11 classical (`Cl`) and 16 improvising (`Imp`) musicians. There are 27,663 standards, 2,615 function deviants and 2,657 exemplar deviants: about 84% standards and 16% deviants.

**Denominators matter.** `signal_pass=False` also describes events whose signal was never assessed. In particular, the three participants failing the initial channel cap did not proceed to epoch QC. Their losses are not evidence that every one of their epochs exceeded 150 µV. Within participants that reached epoch QC, the retained fractions are 97.66% for standards, 97.94% for function deviants and 98.12% for exemplar deviants.

Label-blind rules need not remove equal proportions of each class. The edge guard excludes 13.49% of source-available standards, 10.58% of function deviants and 7.35% of exemplar deviants. Event scheduling matters: the released blocks begin with standards. Neither similar final signal-QC rates nor the absence of labels in the cleaning code proves that exclusions are unrelated to condition, participant group or signal characteristics.

## Table and audit fields

| Field | Role |
| --- | --- |
| `event_id` | Unique join key formed from `subject` and one-based `trial` |
| `subject` | Participant ID and future validation group |
| `trial`, `block`, `onset_s` | One-based recording order and block index, plus middle chord time in seconds from recording start |
| `condition` | `standard`, `function_dev`, or `exemplar_dev` for descriptive reporting |
| `target` | Binary stimulus outcome, 0 for standard and 1 for either deviant |
| `group` | `Cl` = classical, `Imp` = improvising; descriptive metadata from `participants.tsv`, present in the feature table and participant audit |
| `source_available` | Source-eligible participant with enough recorded samples for the whole intended epoch |
| `erp_*_uv` | The 21 allowed EEG predictors |

`events.parquet` contains the retained rows. `event_ledger.csv.gz` contains all candidates, plus `boundary_eligible`, `signal_pass`, and `included` flags. `signal_pass` is false when signal was never assessed. A blank `reason` denotes inclusion. Exclusions distinguish `source_severely_truncated`, `source_unavailable`, `block_boundary_overlap`, `filter_edge_guard`, `signal:*`, and `participant:*`. A signal reason can combine `nonfinite`, `amplitude`, and `flat`. `max_ptp_uv` and `worst_channel` summarize the largest peak-to-peak channel when measured. `participant_audit.csv` records the participant decision separately even when individual epochs also have a signal exclusion.

The participant audit gives source coverage, counts at each stage (`n_candidates`, `n_source_available`, `n_edge_eligible`, `n_signal_pass`, and `n_included`), and signal retention (`n_signal_pass / n_edge_eligible` where the denominator is positive). Read this file and `provenance.json` for the released cohort’s counts. The build checks that feature and included ledger IDs match, IDs are unique, metadata agree with the source, predictors are finite, and every excluded candidate has a reason.

## Reproduction and inspection

The README gives platform-specific setup. The strict reproduction route is `uv sync --locked --extra preprocessing`, followed by `uv run --locked --extra preprocessing python -m eeg_pipeline download` and `uv run --locked --extra preprocessing python -m eeg_pipeline build`. Use `--data-dir` for an existing download. A subset requires its own `--output-dir` so it does not replace the full cohort accidentally. The `audit` command verifies source bytes and event metadata without fitting ICA.

Completed participant results live in the ignored `.cache/` beneath the output directory. Reuse requires matching source, processing-code, configuration, lockfile and runtime identities, plus intact cached output hashes. Changing the platform or package versions can legitimately create a different identity. A fixed random seed and locked dependencies aid reproduction, but do not promise bit-identical floating-point results on every operating system. Local execution has been checked on macOS and text-checkout behavior simulated with Windows-style Git settings; native Windows/Linux runtime checks are provided by the CI workflow and must be confirmed from its actual runs. CI exercises the bundled-data notebook and focused tests, not a full raw-EEG rebuild. The published output hashes identify the specific bundled files; `.gitattributes` preserves their text line endings across checkouts.

The build publishes a completion marker (`provenance.json`) last. Its output checksums must agree before a generation is treated as complete. This avoids accepting a mixture of old and new output files after an interrupted publication. These integrity checks establish file consistency; they do not validate the biological interpretation of the signals.

A rebuild creates `results/.build.lock` containing its process ID so two builds do not write to the same directory. Normal completion and Ctrl-C remove this lock. If a build reports an existing `.build.lock` after a crash or forced termination, confirm that the process ID recorded inside the file is no longer running before removing the stale lock and restarting. Never remove the lock from an active build.

The worked example in `example.npz` includes native-grid raw voltage, aligned analysis voltage, and the corresponding baseline-corrected trace. Waveform arrays are in microvolts and indexed by example, channel, and time. Separate broadband `pre_ica_uv` and `post_ica_uv` arrays show the effect of component removal for the events named by `ica_event_ids`. The selected traces include the first retained event and the first and highest-amplitude eligible event before ICA in each block. Some diagnostic examples are rejected events, so check their `included` flags before treating them as feature-table examples. Comparing traces shows what component removal changed in the recorded signal. Artifact removal can also alter neural activity.

## Residual artifacts and descriptive summaries

The fixed QC screen leaves 215 retained rows with at least one feature above 30 µV in magnitude, including 48 above 50 µV. Of those 48, 41 come from `sub-036`. A baseline-relative window mean can be large even when the whole epoch stays below 150 µV peak-to-peak. Slow variation and residual ocular, muscle or electrode activity are possible explanations; the summary statistics alone do not diagnose the source. Passing the cutoff does not establish that the event is clean.

Do not trim observations merely to improve a model's score. Report the current exclusions and, during modeling, assess whether conclusions depend strongly on extreme events or individual participants. Scaling changes feature units and regularization constrains coefficients; neither removes artifacts or guarantees robustness to influential observations. Any alternative filtering or robust modeling choice belongs inside a documented evaluation plan rather than being chosen on the held-out result.

The condition-summary plots first average within each participant and condition, then average those participant means equally. Their error bars are one standard error across participants, not trial-to-trial spread, 95% confidence intervals, or confidence intervals for paired condition differences. The plots summarize three time windows rather than a continuous ERP waveform. They are descriptive evidence of differences in the processed data, not proof of successful classification or exact event alignment.

## Future evaluation and scope

Use only `PREDICTORS` as model input. Participant IDs, trial order, block, onset, condition, the target itself, source/QC information, and group are metadata or labels. In the released event files, each of the 1,440 trial numbers maps to exactly one condition across all 40 participants. This is an observed property of the files, not a claim that all experimental randomization was absent. Trial position and an event ID containing it can therefore reveal the label even across held-out participants. A future evaluation will group folds by `subject`, with scaling, feature selection, tuning, and other model-side fitting confined to training folds. Report class balance and performance against a class-prior baseline. Participant-grouped evaluation on this dataset measures transfer to other participants under the same task and sequence design.

For an unseen participant, this pipeline uses that person's **complete unlabeled recording** to estimate their own channel repairs, reference, ICA correction and QC. No such estimate is pooled across training and test participants, but access to the complete new recording is still an offline batch assumption. It is stronger than predicting immediately from one arriving event. Zero-phase filtering can also mix samples from later within the block into an earlier baseline or feature, including activity from neighboring chords or trials.

During modeling, split by participant before fitting any shared scaling, tuning, feature selection, class-balancing or threshold-selection procedure. Held-out predictions used for evaluation must not also choose those procedures. Uncertainty calculations should respect participant clustering; event rows and cross-validation folds cannot be treated as independent replicates. Define whether the reported metric averages equally over participants or pools their events, and distinguish uncertainty conditional on fixed predictions from uncertainty that includes refitting the model.

This milestone provides a prepared dataset and descriptive exploration. Because this dataset has already been explored, later cross-validation results should be read as exploratory rather than as a final, untouched test. Generalization to new acoustic material, a new task, or real-time deployment would require evaluation designed for those settings.

## References

- **Dataset:** Goldman, A., Jackson, T., & Sajda, P. (2026). *EEG: Improvisation and Musical Structures* (Version 1.0.0) [Dataset]. NEMAR `on003570`, https://doi.org/10.82901/nemar.on003570 (mirroring OpenNeuro `ds003570`, https://doi.org/10.18112/openneuro.ds003570.v1.0.0).
- **Original EEG study:** Faller, J., Goldman, A., Lin, Y., McIntosh, J. R., & Sajda, P. (2021). Spatiospectral brain networks reflective of improvisational experience. *NeuroImage*, 242, 118458. https://doi.org/10.1016/j.neuroimage.2021.118458
- **Behavioral background study:** Goldman, A., Jackson, T., & Sajda, P. (2020). Improvisation experience predicts how musicians categorize musical structures. *Psychology of Music*, 48(1), 18–34. https://doi.org/10.1177/0305735618779444
- **EEG software and methods:** Gramfort, A., et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267, https://doi.org/10.3389/fnins.2013.00267; Bigdely-Shamlo, N., et al. (2015). The PREP pipeline: standardized preprocessing for large-scale EEG analysis. *Frontiers in Neuroinformatics*, 9, 16, https://doi.org/10.3389/fninf.2015.00016 (`PyPREP`); Ablin, P., Cardoso, J.-F., & Gramfort, A. (2018). Faster independent component analysis by preconditioning with Hessian approximations. *IEEE Transactions on Signal Processing*, 66(15), 4040–4049, https://doi.org/10.1109/TSP.2018.2844203 (`python-picard`); Pion-Tonachini, L., Kreutz-Delgado, K., & Makeig, S. (2019). ICLabel: An automated electroencephalographic independent component classifier, dataset, and website. *NeuroImage*, 198, 181–197, https://doi.org/10.1016/j.neuroimage.2019.05.026; Li, A., et al. (2022). MNE-ICALabel: Automatically annotating ICA components with ICLabel in Python. *Journal of Open Source Software*, 7(76), 4484, https://doi.org/10.21105/joss.04484.
