# SECOM Wafer Fault Prediction

Prediction of which semiconductor wafers fail the final electrical test, from
590 anonymised sensor readings of 1,567 wafers (SECOM dataset, about 7 percent
failures). The analysis follows CRISP-DM: stratified train/test split, cleaning,
3σ outlier treatment, kNN imputation, Boruta feature selection, balancing and
classification models, with the final model chosen by total cost
(200 € per false alarm, 5,000 € per missed defect).

Case study for Advanced Data Analytics and Computational Methods
(MPMD, HTW Berlin, winter semester 2026/27).

## Team
- Revathy Harichandran
- Varshini
- Gul

This version of the code was written by **Revathy Harichandran**.

## Requirements
- **Python 3.14** (developed and tested with Python 3.14.7). Python 3.12 or newer
  is the minimum; the script stops with a message on older versions.
- The exact package versions are pinned in `requirements.txt`.
- An internet connection: the data is downloaded from the UCI repository at every run.

## Setup

Open a terminal in the project folder.

**Windows (PowerShell)**
```
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Mac (Terminal)**
```
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The first line creates a virtual environment (a private copy of Python in the
`.venv` folder); the second installs the pinned package versions into it.

The script also installs any missing or differently versioned package itself
when it starts. Running it inside a fresh virtual environment is recommended,
so that it does not change package versions of another Python installation.

## Running the script in VS Code
1. Install the **Python** extension (by Microsoft).
2. Open the project folder (**File → Open Folder**).
3. Press **Ctrl+Shift+P** (Mac: **Cmd+Shift+P**), choose
   **Python: Select Interpreter** and select the one in `.venv`.
4. Open `src/secom_analysis.py` and click **Run Python File** (▶, top right).

The script is divided into numbered blocks starting with `# %%`. They can also be
run one at a time with **Run Cell**; VS Code then asks to install the `ipykernel`
package once, which is only needed for this interactive mode.

## Outputs
- Figures: `results/figures/`
- Tables: `results/tables/`

Both folders are created by the script if they do not exist. The data itself is
never saved.

## Data
McCann, M. and Johnston, A. (2008). SECOM [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C54305

## Licence
The code in this repository is released under the MIT licence (see `LICENSE`).
The SECOM dataset is **not** covered by this licence: it belongs to its donors
and is licensed under Creative Commons Attribution 4.0 (CC BY 4.0), which allows
use with citation. The dataset is not stored in this repository; the script
downloads it from UCI at runtime.
