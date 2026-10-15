# %% Block 0: Header, package installation and imports
# SECOM wafer fault prediction
# Case study for Advanced Data Analytics and Computational Methods (MPMD, HTW Berlin)
# Team: Revathy Harichandran, Varshini, Gul
#
# Predicts which semiconductor wafers fail the final electrical test from
# 590 sensor readings, following CRISP-DM and the SECOM Case Study Manual.
# The best model is chosen by total cost: 200 euros per false positive and
# 5,000 euros per false negative.
#
# Data: McCann, M. and Johnston, A. (2008). SECOM [Dataset]. UCI Machine
# Learning Repository, CC BY 4.0. https://doi.org/10.24432/C54305
# The data is downloaded from UCI every time the script runs; no local copy is used.
#
# How to run: open this file in Visual Studio Code and click "Run Python File",
# or run it block by block with the "# %%" cells. The script installs its own
# packages (below), so it reproduces every result without any modification.
#
# This block installs the exact package versions the team uses, because the
# professor requires the script to install its own packages (manual p. 4, slide 7)
# and identical versions give identical results on every computer.

# Standard-library modules: they come with Python, so they never need installing.
import sys
import subprocess
import importlib
import importlib.metadata

# Stop early with a clear message on Python older than 3.12, because the pinned
# numpy 2.5.4 and scipy 1.18.1 cannot be installed there (pip would fail with a
# long, confusing error instead). The team uses Python 3.14.
if sys.version_info < (3, 12):
    print("This script needs Python 3.12 or newer. You are using Python", sys.version.split()[0])
    sys.exit(1)

# Exact versions, copied from requirements.txt. They are written here as well
# because the professor receives only this script, not requirements.txt.
# If requirements.txt changes, this list must be changed in the same way.
REQUIRED_PACKAGES = {
    "Boruta": "0.4.3",
    "cloudpickle": "3.1.2",
    "contourpy": "1.4.0",
    "cycler": "0.12.1",
    "fonttools": "4.67.0",
    "imbalanced-learn": "0.14.2",
    "joblib": "1.6.0",
    "kiwisolver": "1.5.1",
    "matplotlib": "3.11.2",
    "narwhals": "2.27.1",
    "numpy": "2.5.4",
    "packaging": "26.3",
    "pandas": "3.0.6",
    "pillow": "12.3.0",
    "pyparsing": "3.3.3",
    "python-dateutil": "2.9.0.post0",
    "scikit-learn": "1.9.1",
    "scipy": "1.18.1",
    "six": "1.17.0",
    "sklearn-compat": "0.1.6",
    "threadpoolctl": "3.7.0",
    "tzdata": "2026.5",
}


def find_packages_to_install(required_packages):
    """
    Compare the installed version of every required package with the pinned version.

    Input:  required_packages (dict: package name -> exact version)
    Output: list of "name==version" strings for every package that is missing
            or installed in a different version
    """
    packages_to_install = []
    for package_name, required_version in required_packages.items():
        # version() reads the installed version from the package's metadata,
        # without importing the package. It raises PackageNotFoundError if the
        # package is not installed at all.
        try:
            installed_version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            installed_version = None

        # A missing package and a different version both need installing.
        if installed_version != required_version:
            packages_to_install.append(package_name + "==" + required_version)
    return packages_to_install


packages_to_install = find_packages_to_install(REQUIRED_PACKAGES)

if len(packages_to_install) > 0:
    print("Installing", len(packages_to_install), "package(s):", packages_to_install)
    # sys.executable is the Python running this script, so "-m pip" installs into
    # exactly this Python, not into another Python on the same computer.
    # check_call stops the script with an error if pip fails.
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages_to_install)
    # Python remembers which packages existed when it started; this makes it
    # look again, so the packages installed just now can be imported below.
    importlib.invalidate_caches()
else:
    print("All", len(REQUIRED_PACKAGES), "packages already have the right versions.")

# Imports used by the blocks written so far. Later blocks add their imports
# (numpy, matplotlib, scikit-learn, imbalanced-learn, Boruta) when first needed.
from pathlib import Path
import pandas as pd


# %% Block 1: Settings
# Every fixed number and path of the project is defined here once, so it can be
# found and changed in one place and is used identically by every later block.

# Random seed, used in the split, the balancing, every model and the
# cross-validation splitter, so that every run and every laptop gives the same results.
SEED = 42

# Cost of each error type in euros (manual Section 14.2): a false positive is a
# good wafer wrongly flagged, a false negative is a failed wafer that is missed.
COST_FALSE_POSITIVE = 200
COST_FALSE_NEGATIVE = 5000

# Share of wafers held back as the test set: the 80/20 split from manual Section 3.
TEST_SIZE = 0.2

# Download links of the two original files in the UCI repository (dataset 179).
URL_SENSOR_DATA = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
URL_LABELS = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"

# Find the project folder from the script's own location, so that the paths
# work on every laptop. __file__ is the path of this script (inside src), so the
# project folder is one level up. In VS Code's interactive window __file__ does
# not exist; there the working folder is the script's folder (src) by default.
if "__file__" in globals():
    SCRIPT_FOLDER = Path(__file__).resolve().parent
else:
    SCRIPT_FOLDER = Path.cwd()
PROJECT_FOLDER = SCRIPT_FOLDER.parent

# Output folders for figures and tables. mkdir creates them if they are missing
# (for example on the professor's computer); parents=True also creates "results",
# and exist_ok=True means no error if the folder already exists.
FIGURES_FOLDER = PROJECT_FOLDER / "results" / "figures"
TABLES_FOLDER = PROJECT_FOLDER / "results" / "tables"
FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
TABLES_FOLDER.mkdir(parents=True, exist_ok=True)

print("Figures are saved to:", FIGURES_FOLDER)
print("Tables are saved to: ", TABLES_FOLDER)


# %% Block 2: Download the data
# Reads both original files directly from the UCI repository at runtime, because
# the professor requires the script to download the data itself (manual p. 4,
# slide 7). Column names, label coding and time stamps follow in Phase 1.

# Both files separate values with a single space (sep=" "; the pandas default
# would be a comma) and have no header row (header=None; by default pandas would
# use the first wafer as column names). pandas reads the text "NaN" as a missing
# value by default; missing values are flagged explicitly in Phase 1.
sensor_data = pd.read_csv(URL_SENSOR_DATA, sep=" ", header=None)
labels = pd.read_csv(URL_LABELS, sep=" ", header=None)

# Expected: (1567, 590) for the sensors and (1567, 2) for label and time stamp.
print("Sensor data shape:", sensor_data.shape)
print("Labels shape:     ", labels.shape)


# %% Block 3: Data understanding
# Shape, types, duplicates, class balance, statistics, missingness and figures (Phase 1).

# %% Block 4: Stratified train/test split
# Stratified 80/20 split with SEED, and a check of the class shares (Phase 1).

# %% Block 5: Rough feature reduction
# Remove constant, duplicate and very incomplete columns on the training set, mirrored to the test set (Phase 1).

# %% Block 6: Correlation heatmap
# Visualise correlations between the remaining sensors, without removing any (Phase 1).

# %% Block 7: Outliers (3 sigma rule)
# Detect outliers with the 3 sigma rule and apply the chosen treatment (Phase 1).

# %% Block 8: Scaling and kNN imputation
# Scale the features, then fill missing values with kNN imputation fitted on the training set (Phase 2).

# %% Block 9: First random forest
# Train a first random forest and report its metrics (Phase 2).

# %% Block 10: Boruta feature selection
# Select the informative sensors with Boruta on the training set (Phase 2).

# %% Block 11: Balancing
# Balance the training set with SMOTE or ROSE; the test set is never balanced (Phase 2).

# %% Block 12: Evaluation helpers
# Functions for the confusion matrix, metrics, ROC curve and total cost (Phase 2).

# %% Block 13: Full pipeline
# Put all preprocessing and the model into one pipeline to avoid leakage in cross-validation (Phase 3).

# %% Block 14: Model comparison
# Compare the random forest with at least one other model using stratified cross-validation (Phase 3).

# %% Block 15: Grid search
# Tune the settings of preprocessing and models with a small grid search (Phase 3).

# %% Block 16: Threshold and cost optimisation
# Choose the decision threshold with the lowest cost on cross-validated training predictions (Phase 3).

# %% Block 17: Final evaluation on the test set
# Evaluate the chosen model once on the untouched test set (Phase 3).

# %% Block 18: Save results and summary
# Save all tables and figures and print a summary of the results (Phase 3).
