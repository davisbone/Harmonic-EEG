# Harmonic-EEG

## Set up the environment

From a terminal, check for `uv` with `uv --version`. If you need to install it, use the command for your operating system and then open a new terminal. [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)

macOS or Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

From the repository root, set up the locked environment and open the milestone notebook.

```text
uv sync
uv run jupyter lab ms2_data.ipynb
```

`uv sync` creates a `.venv` folder with the exact package versions listed in `uv.lock` (Python 3.12), so everyone works in the same environment. The `.venv` folder is ignored by git. In VS Code, select it with **Python: Select Interpreter** (or the kernel picker in a notebook) and choose `.venv`.

- **After a `git pull`:** run `uv sync` again to pick up any dependency changes.
- **Adding a package:** run `uv add <package>`, then commit both `pyproject.toml` and `uv.lock`.
- **Running things from the terminal:** prefix commands with `uv run`, for example `uv run jupyter lab`.

## Explore the prepared data

**Can 21 early EEG voltage features tell standard from deviant middle chords in participants the model was not trained on?** This CS1090A Group 18 milestone prepares and describes an event-level table for that offline classification question. Each retained event has 21 EEG voltage features. The binary label pools the two deviant types, which remain identifiable in the `condition` column.

`ms2_data.ipynb` reads the bundled `results/` files and runs without a raw-data download or the optional preprocessing packages (`A1exploreData.ipynb` preserves earlier exploration of `participants.tsv`). It shows the current cohort, table schema, class balance, trial exclusions, and a worked voltage-feature example. The [visual walkthrough](docs/explainer.html) opens locally in a browser. The [methods](docs/methods.md) specify the pipeline, field roles, and interpretation limits.

| Bundled file | Contents |
| --- | --- |
| `results/events.parquet` | One row per retained event with the fixed predictor columns and metadata |
| `results/event_ledger.csv.gz` | One row per candidate event with eligibility, QC, and exclusion reasons |
| `results/participant_audit.csv` | Source coverage, channel repairs, ICA decisions, and event counts by participant |
| `results/example.npz` | A small set of waveform traces for the worked example |
| `results/provenance.json` | Dataset version, runtime, settings, predictor names, cohort counts, and output hashes |

## Rebuild from raw EEG

Use the pinned NEMAR `on003570` v1.0.0 release. The rebuild manifest covers about 39 GB of inputs. Reserve time for downloading, checksum validation, and sequential CPU processing. A full build can take hours. The notebook quickstart uses the prepared files above. [NEMAR dataset record](https://nemar.org/dataset/on003570)

```text
uv sync --extra preprocessing
uv run --extra preprocessing python -m eeg_pipeline download
uv run --extra preprocessing python -m eeg_pipeline build
```

The default input is `Data/on003570` and the default build destination is `results`. The `Data` folder is ignored by git, so the raw recordings never get committed. If you already have the release, point `build` at the directory containing `participants.tsv` and the `sub-*` folders. `download` uses the same `--data-dir` option and verifies files already there. Replace the example path with your own.

```text
uv run --extra preprocessing python -m eeg_pipeline build --data-dir "path/to/on003570"
```

To verify source coverage without fitting ICA, run `uv run --extra preprocessing python -m eeg_pipeline audit` with the same input options. For a small rebuild, download just the selected participant files and use a separate output directory.

```text
uv run --extra preprocessing python -m eeg_pipeline download --subjects sub-002
uv run --extra preprocessing python -m eeg_pipeline build --subjects sub-002 --output-dir Data/sub-002-results
```

Rerunning `build` skips participants that already finished, so an interrupted build can be restarted.

The source release is CC0. Cite the [dataset](https://doi.org/10.82901/nemar.on003570) and the [original EEG paper](https://doi.org/10.1016/j.neuroimage.2021.118458) when using these data.
