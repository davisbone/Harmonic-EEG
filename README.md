# Harmonic-EEG

### Setting Up Your Environment
This project uses [uv](https://docs.astral.sh/uv/). Check whether it's installed:

```bash
uv --version
```

If that prints a version number, skip ahead. Otherwise, install uv and then open a new terminal:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then, from the root of the repository, run:

```bash
uv sync
```

This creates a `.venv` folder with the exact package versions listed in `uv.lock`, so everyone works in the same environment. The `.venv` folder is ignored by git. In VS Code, select it with **Python: Select Interpreter** (or the kernel picker in a notebook) and choose `.venv`.

- **After a `git pull`:** run `uv sync` again to pick up any dependency changes.
- **Adding a package:** run `uv add <package>` (for example, `uv add mne`), then commit both `pyproject.toml` and `uv.lock`.
- **Running things from the terminal:** prefix commands with `uv run`, for example `uv run jupyter lab`.

### Downloading Data
Data can be found and installed at https://www.nemar.org/dataset/on003570. You can download the data and store it in the repository in a folder titled "Data" (without the quotations) and it will be ignored from git, as I've included it in the .gitignore file. Make sure to have your data folder ignored by git or use a different folder entirely that is outside of this repository. 