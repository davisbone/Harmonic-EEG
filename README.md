# Harmonic-EEG

**[Open the visual walkthrough in your browser](https://html-preview.github.io/?url=https://github.com/davisbone/Harmonic-EEG/blob/ms2-lean/docs/explainer.html)** (works on phones)

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

## What's here

**Question:** Can 21 early EEG voltage features tell standard from deviant middle chords in participants the model was not trained on?

| File | Contents |
| --- | --- |
| `ms2_data.ipynb` | The Milestone 2 notebook: getting the data, what a row means, each cleaning step demonstrated on one participant, and the refined question |
| `preprocess.py` | All the EEG cleaning in one file (about 430 lines): raw recordings in, one row per middle chord out |
| `data_manifest.json` | Every file in the NEMAR release with its size and download link |
| `results/events.parquet` | The table: one row per retained event, 21 EEG features plus metadata |
| `results/event_ledger.csv.gz` | All 57,600 candidate events, with the reason each was kept or removed |
| `results/participant_audit.csv` | Per participant: recording coverage, bad electrodes, and event counts |
| `A1exploreData.ipynb` | Earlier exploration of `participants.tsv` |

The notebook runs without the raw EEG: it loads the table from `results/` (or from GitHub if only the notebook is downloaded). The [visual walkthrough](docs/explainer.html) and [methods](docs/methods.md) explain the processing in more detail.

## Rebuild from raw EEG (optional)

This downloads about 39 GB and takes roughly 75 minutes of processing. The `Data` folder is ignored by git.

```text
uv sync --extra preprocessing
uv run --extra preprocessing python -c "import preprocess; preprocess.download('Data/on003570')"
uv run --extra preprocessing python preprocess.py Data/on003570 results
```

To try one participant, download only its files and write to a separate folder:

```text
uv run --extra preprocessing python -c "import preprocess; preprocess.download('Data/on003570', ['sub-002'])"
uv run --extra preprocessing python preprocess.py Data/on003570 Data/sub-002-results sub-002
```

The source release is CC0. Cite the [dataset](https://doi.org/10.82901/nemar.on003570) and the [original EEG paper](https://doi.org/10.1016/j.neuroimage.2021.118458) when using these data.
