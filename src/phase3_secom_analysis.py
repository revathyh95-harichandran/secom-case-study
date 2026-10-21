# %% Block 0: Header, package installation and imports
# SECOM wafer fault prediction - PHASE 3 SCRIPT (Part C, Presentation 3, final submission)
# Blocks 0 to 7 (Phase 1, unchanged): installation, settings, data download,
# data understanding, stratified split, feature reduction, correlation, outliers.
# Blocks 8 to 12 (Phase 2, unchanged): missing-value check, scaling and kNN
# imputation, first random forest, Boruta feature selection, balancing, evaluation.
# Blocks 13 to 18 (Phase 3): leak-proof pipeline, model comparison, grid search,
# cost-based threshold, the one-time final test, summary. Each phase has
# its own complete script (phase1, phase2, phase3); every script runs on its own
# from the download onwards, so it can be submitted and run without the others.
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
# The data is downloaded from UCI every time the script runs; no local copy is
# read. The downloaded files are saved unchanged in data/raw, and every prepared
# version of the data is saved in data/processed (neither folder is on GitHub).
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
# (more of scikit-learn, imbalanced-learn, Boruta) when first needed.
import io
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
# The script only saves figures to files and never opens a window, so matplotlib
# uses the file-only "Agg" backend. The default window backend (Tk) printed
# "main thread is not in main loop" errors when a multi-threaded step ran
# (DEBUG_LOG.md, 20 Oct). use() must come before pyplot is imported.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split
# Phase 2 imports: scaling and kNN imputation (Block 8).
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
# Phase 2 imports: first model and cross-validation (Block 9).
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
# Phase 2 import: Boruta feature selection (Block 10).
from boruta import BorutaPy
# Phase 2 imports: balancing (Block 11).
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from sklearn.neighbors import NearestNeighbors
# Phase 2 imports: evaluation (Block 12).
from sklearn.metrics import roc_auc_score, roc_curve
# Phase 3 imports: the leak-proof pipeline (Block 13).
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import clone
# Phase 3 imports: model comparison (Block 14).
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
# Phase 3 imports: grid search and timing (Block 15).
import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, fbeta_score
# Phase 3 import: precision-recall curves (Block 16).
from sklearn.metrics import precision_recall_curve
# Phase 3 imports: saving the final pipeline and the reproducibility record (Block 18).
import joblib
import hashlib
import platform
import datetime


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

# Missing-value threshold in percent (decision D5): columns with MORE missing
# values than this in the training set are removed in Block 5. 60 % is the
# manual's starting point; on the training set any threshold from 49.1 % to
# 61.2 %, including the papers' 55 %, removes exactly the same 24 columns.
MISSING_THRESHOLD = 60

# Correlation limit used only to COUNT strongly correlated pairs (|r| > 0.8, the
# value in manual 4.6 and the papers). No column is removed because of it.
CORRELATION_LIMIT = 0.8

# Outliers (manual Section 5): a value more than OUTLIER_SIGMA standard
# deviations from its column's training mean is an outlier (3 sigma rule; the
# IQR method is not allowed, rule 4). OUTLIER_TREATMENT (decision D7) is one of
# "nothing", "cap" or "remove"; "cap" keeps every wafer and value but pulls
# extreme values back to the bound. All three stay available for the grid search.
OUTLIER_SIGMA = 3
OUTLIER_TREATMENT = "cap"

# Phase 2 setting - kNN imputation (decision D9): a gap is filled with the
# average of the KNN_NEIGHBORS most similar training wafers. 5 is the manual's
# required starting point; it stays a setting so the grid search can try 3 to 9.
KNN_NEIGHBORS = 5

# Phase 2 setting - cross-validation: the training set is cut into CV_FOLDS
# parts (stratified, so each part keeps about 6.6 % failures). 5 is the manual's
# choice and leaves about 17 failed wafers in each part to evaluate on.
CV_FOLDS = 5

# Phase 2 settings - Boruta feature selection: BORUTA_MAX_ITER is the maximum
# number of rounds against the shadow features (100 is Boruta's default; 200
# gave the same 15 confirmed sensors). KEEP_TENTATIVE (decision D10): True keeps
# the tentative (borderline) sensors, as the manual's week 4 asks
# ("including tentative ones").
BORUTA_MAX_ITER = 100
KEEP_TENTATIVE = True

# Phase 2 settings - balancing (decision D11). BALANCING_METHOD is the working
# choice for Phase 2 after the screening in Block 11 ("none", "class_weight",
# "SMOTE", "ROSE" or "ADASYN"): ROSE had the lowest cross-validated cost.
# ROSE_SHRINKAGE = 1.0 reproduces ROSE's default amount of smoothing (the noise
# added around each copied failed wafer); 0 would mean plain copies.
BALANCING_METHOD = "ROSE"
ROSE_SHRINKAGE = 1.0

# Download links of the two original files in the UCI repository (dataset 179).
URL_SENSOR_DATA = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
URL_LABELS = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"

# Number of wafers documented by UCI and the manual (Section 2.1); Block 2 checks
# that both downloaded files have exactly this many rows.
EXPECTED_ROWS = 1567

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

# Data folders (standard practice): data/raw keeps the downloaded files exactly
# as published, data/processed keeps every prepared version of the data. Both
# are created if missing and are excluded from GitHub by .gitignore.
RAW_DATA_FOLDER = PROJECT_FOLDER / "data" / "raw"
PROCESSED_DATA_FOLDER = PROJECT_FOLDER / "data" / "processed"
RAW_DATA_FOLDER.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_FOLDER.mkdir(parents=True, exist_ok=True)

print("Figures are saved to:", FIGURES_FOLDER)
print("Tables are saved to: ", TABLES_FOLDER)
print("Raw data is saved to:", RAW_DATA_FOLDER)
print("Processed data is saved to:", PROCESSED_DATA_FOLDER)

# Colours used in every figure, defined once so all charts look the same.
# Blue and orange are a colour-blind-safe pair; grey is for reference lines.
COLOR_PASS = "#2a78d6"
COLOR_FAIL = "#eb6834"
COLOR_MAIN = "#2a78d6"
COLOR_REFERENCE = "#52514e"


# %% Block 2: Download and prepare the data
# Reads both original files directly from the UCI repository at runtime with
# pandas (team decision D1), because the professor requires the script to
# download the data itself (manual p. 4, slide 7). Then names the columns,
# parses the time stamps, recodes the label to fail = 1 / pass = 0 (decision D2)
# and keeps features (X) and target (y) as separate objects (manual Section 2.1).
# Nothing is removed or changed in the sensor values here. The two downloaded
# files are saved unchanged in data/raw; the prepared data is saved in data/processed.


def download_file(url, raw_file_name):
    """
    Download a file from the internet and save its bytes unchanged in data/raw.

    Input:  url (str: web address of the file), raw_file_name (str: name in data/raw)
    Output: the downloaded bytes (so the data is read from the download itself,
            never from a local copy)
    """
    # urlopen() requests the file; read() returns its exact bytes. timeout=60
    # stops waiting after 60 seconds if UCI does not answer.
    with urllib.request.urlopen(url, timeout=60) as response:
        file_bytes = response.read()
    # "wb" = write bytes: the raw file is byte-for-byte what UCI published and is
    # rewritten on every run, so it always matches the current download.
    with open(RAW_DATA_FOLDER / raw_file_name, "wb") as raw_file:
        raw_file.write(file_bytes)
    print("Downloaded", len(file_bytes), "bytes and saved", RAW_DATA_FOLDER / raw_file_name)
    return file_bytes


def save_processed(features, target, file_name):
    """
    Save a prepared version of the data as a CSV file in data/processed.

    Input:  features (DataFrame), target (Series with the label, same rows as features),
            file_name (str, for example "block4_train_split.csv")
    Output: none (the file is written and its path printed)
    """
    # Copy so the features in memory are not changed; add the label as the last column.
    table = features.copy()
    table["target"] = target
    # index_label="wafer_id" stores each wafer's original row number, so every
    # row can be traced back to the raw files.
    table.to_csv(PROCESSED_DATA_FOLDER / file_name, index_label="wafer_id")
    print("Saved processed data:", PROCESSED_DATA_FOLDER / file_name, table.shape)


sensor_bytes = download_file(URL_SENSOR_DATA, "secom.data")
label_bytes = download_file(URL_LABELS, "secom_labels.data")

# Read the sensor readings from the downloaded bytes (io.BytesIO lets pandas read
# bytes in memory as if they were a file). Values are separated by a single
# space (sep=" "; the pandas default would be a comma) and there is no header
# row (header=None; by default pandas would use the first wafer as column names).
# na_values=["NaN"] states explicitly that the text "NaN" marks a missing
# reading, so every gap becomes a real missing value (rule 5) instead of
# relying silently on pandas' default list of missing-value markers.
sensor_data = pd.read_csv(io.BytesIO(sensor_bytes), sep=" ", header=None, na_values=["NaN"])

# Read the labels file: column 1 is the label, column 2 the test date and time.
# names= gives the two columns their names directly. The date-time contains a
# space but stays one field, because it is written in double quotes and '"' is
# pandas' default quote character.
labels = pd.read_csv(io.BytesIO(label_bytes), sep=" ", header=None, names=["target", "timestamp"])

# Check that both files describe the same 1567 wafers. The files have no wafer
# ID: row i of the labels belongs to row i of the sensors, so a different row
# count would mean the two files no longer match. The script stops in that case.
print("Rows in sensor file:", len(sensor_data), "| rows in labels file:", len(labels))
if len(sensor_data) != EXPECTED_ROWS or len(labels) != EXPECTED_ROWS:
    raise ValueError("Expected " + str(EXPECTED_ROWS) + " rows in both files.")

# Name the columns sensor_0 to sensor_589, so every sensor keeps a traceable name
# through all later steps (manual Checklist 2.1). The loop builds one name per
# column; range(n) counts from 0 to n - 1.
sensor_names = []
for column_number in range(sensor_data.shape[1]):
    sensor_names.append("sensor_" + str(column_number))
sensor_data.columns = sensor_names
print("Columns:", len(sensor_names), "from", sensor_names[0], "to", sensor_names[-1])

# Turn the time stamps from text into real dates (decision D3: parse and explore).
# The file writes day/month/year (for example "19/07/2008 11:55:00"), so the
# format is given explicitly: without it, a date such as 01/08/2008 could be read
# as 8 January instead of 1 August. A value in any other format raises an error.
labels["timestamp"] = pd.to_datetime(labels["timestamp"], format="%d/%m/%Y %H:%M:%S")
print("Test dates from", labels["timestamp"].min(), "to", labels["timestamp"].max())

# Recode the label (decision D2): UCI uses -1 = pass and 1 = fail; scikit-learn
# treats 1 as the positive class, so pass becomes 0 and fail stays 1.
counts_before = labels["target"].value_counts()
print("Label counts before recoding (-1 = pass, 1 = fail):")
print(counts_before)

# map() replaces each value using the dictionary. Any value not in the
# dictionary would become missing, so zero missing values afterwards proves that
# the file contained only -1 and 1.
labels["target"] = labels["target"].map({-1: 0, 1: 1})
if labels["target"].isna().sum() != 0:
    raise ValueError("The labels file contains values other than -1 and 1.")

counts_after = labels["target"].value_counts()
print("Label counts after recoding (0 = pass, 1 = fail):")
print(counts_after)

# Proof of correct recoding: every former -1 must now be 0, every 1 must still be 1.
if counts_before[-1] != counts_after[0] or counts_before[1] != counts_after[1]:
    raise ValueError("Label counts changed during recoding.")
print("Recoding check passed: pass", counts_before[-1], "->", counts_after[0],
      "| fail", counts_before[1], "->", counts_after[1])

# Keep features and target as separate objects (manual 2.1): X holds only what a
# model may use, y what it must predict, so the label can never slip into the
# features by accident. The time stamps are kept separately for exploration.
X = sensor_data
y = labels["target"]
timestamps = labels["timestamp"]
print("X shape:", X.shape, "| y shape:", y.shape)

# Processed data, step 1: the full labelled data (named sensors, recoded target)
# plus the parsed time stamp, before any split or cleaning.
X_with_timestamp = X.copy()
X_with_timestamp["timestamp"] = timestamps
save_processed(X_with_timestamp, y, "block2_secom_labelled.csv")


# %% Block 3: Data understanding (descriptive analysis of the full data)
# Describes the full data set before the split: structure, class balance, scales,
# variance, missing values, failures over time and correlations (manual Sections
# 2.2 to 2.7; "Descriptive Analysis" in the professor's slides). This block only
# describes: nothing is removed, filled in or fitted, and no threshold is chosen
# here. All cleaning thresholds are decided later on the training set only
# (Block 5), so the test set never influences a decision (rules 1 and 2).


def save_figure(figure, file_name):
    """
    Save a figure as a PNG file in results/figures and close it.

    Input:  figure (matplotlib Figure), file_name (str, for example "block3_01_x.png")
    Output: none (the file is written and its path printed)
    """
    file_path = FIGURES_FOLDER / file_name
    # dpi=150 gives a sharp image for slides; bbox_inches="tight" trims empty margins.
    figure.savefig(file_path, dpi=150, bbox_inches="tight")
    # Closing frees memory and stops figures from piling up or popping up as windows.
    plt.close(figure)
    print("Saved figure:", file_path)


def plot_variance_histogram(column_variances, title, file_name):
    """
    Draw a histogram of column variances on a log10 scale and save it.
    Variances span many orders of magnitude, so log10 is used (-2 = 0.01,
    0 = 1, 2 = 100). The log of 0 does not exist, so zero variances are left out.

    Input:  column_variances (Series: one variance per column), title (str), file_name (str)
    Output: none (the figure is saved)
    """
    nonzero_variances = column_variances[column_variances > 0]
    log_variances = np.log10(nonzero_variances)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.hist(log_variances, bins=40, color=COLOR_MAIN, edgecolor="white")
    axis.set_xlabel("Column variance, log10 scale (-2 = 0.01, 0 = 1, 2 = 100)")
    axis.set_ylabel("Number of columns")
    axis.set_title(title)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    save_figure(figure, file_name)


def plot_variance_pareto(column_variances, title, file_name):
    """
    Draw a Pareto chart of variance thresholds (x, log scale) against the number
    of columns with a variance at or below each threshold (y), and save it.
    np.logspace(-8, 8, 33) gives 33 thresholds from 10^-8 to 10^8, two per
    power of ten, covering the range of SECOM's variances.

    Input:  column_variances (Series: one variance per column), title (str), file_name (str)
    Output: DataFrame with the columns variance_threshold and columns_at_or_below
    """
    variance_thresholds = np.logspace(-8, 8, 33)
    columns_at_or_below = []
    for threshold in variance_thresholds:
        columns_at_or_below.append((column_variances <= threshold).sum())

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(variance_thresholds, columns_at_or_below, marker="o", markersize=4,
              linewidth=2, color=COLOR_MAIN)
    axis.set_xscale("log")
    # A reference line for the zero-variance columns, only if there are any.
    zero_count = (column_variances == 0).sum()
    if zero_count > 0:
        axis.axhline(zero_count, linestyle="--", color=COLOR_REFERENCE)
        axis.text(variance_thresholds[-1], zero_count + 10, str(zero_count) + " zero-variance columns ",
                  ha="right", va="bottom", color=COLOR_REFERENCE)
    axis.set_xlabel("Variance threshold (log scale)")
    axis.set_ylabel("Columns with variance at or below threshold")
    axis.set_title(title)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    save_figure(figure, file_name)

    return pd.DataFrame({"variance_threshold": variance_thresholds,
                         "columns_at_or_below": columns_at_or_below})


def plot_missing_per_column(missing_percent_per_column, title, file_name):
    """
    Draw the manual's missing-value "histogram": one bar per column (x = column,
    y = % missing), sorted from most to least missing, with reference lines at
    55 % (papers) and 60 % (manual), and save it.

    Input:  missing_percent_per_column (Series: % missing per column), title (str), file_name (str)
    Output: none (the figure is saved)
    """
    missing_percent_sorted = missing_percent_per_column.sort_values(ascending=False)
    figure, axis = plt.subplots(figsize=(10, 4))
    column_positions = range(len(missing_percent_sorted))
    axis.bar(column_positions, missing_percent_sorted.values, width=1.0, color=COLOR_MAIN)
    for reference_percent in [55, 60]:
        axis.axhline(reference_percent, linestyle="--", linewidth=1, color=COLOR_REFERENCE)
    axis.text(len(missing_percent_sorted), 55, "55 % (papers) ", ha="right", va="top", color=COLOR_REFERENCE)
    axis.text(len(missing_percent_sorted), 60, "60 % (manual) ", ha="right", va="bottom", color=COLOR_REFERENCE)
    axis.set_xlabel("Sensor columns, sorted from most to least missing ("
                    + str(len(missing_percent_sorted)) + " columns)")
    axis.set_ylabel("% missing")
    axis.set_title(title)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    save_figure(figure, file_name)


def plot_missing_pareto(missing_percent_per_column, title, file_name):
    """
    Draw a Pareto chart of missing-value thresholds 0, 5, ..., 100 % (x) against
    the number of columns with MORE missing values than the threshold, i.e. the
    columns that would be removed (y), and save it.

    Input:  missing_percent_per_column (Series: % missing per column), title (str), file_name (str)
    Output: DataFrame with the columns threshold_percent and columns_removed
    """
    missing_thresholds = list(range(0, 101, 5))
    columns_removed = []
    for threshold in missing_thresholds:
        columns_removed.append((missing_percent_per_column > threshold).sum())

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(missing_thresholds, columns_removed, marker="o", markersize=5, linewidth=2, color=COLOR_MAIN)
    for reference_percent in [55, 60]:
        axis.axvline(reference_percent, linestyle="--", linewidth=1, color=COLOR_REFERENCE)
    axis.text(55, max(columns_removed), "55 % ", ha="right", va="top", color=COLOR_REFERENCE)
    axis.text(60, max(columns_removed), " 60 %", ha="left", va="top", color=COLOR_REFERENCE)
    axis.set_xlabel("Missing-value threshold (%)")
    axis.set_ylabel("Columns that would be removed")
    axis.set_title(title)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    save_figure(figure, file_name)

    return pd.DataFrame({"threshold_percent": missing_thresholds,
                         "columns_removed": columns_removed})


def find_strong_pairs(correlation_matrix, limit):
    """
    List every pair of columns whose absolute correlation is above a limit.
    The loop visits each pair once (j starts after i, so a column is never
    paired with itself); a missing r (not enough shared wafers) never counts.

    Input:  correlation_matrix (DataFrame: square matrix of r values), limit (float, e.g. 0.8)
    Output: DataFrame with sensor_a, sensor_b and r, strongest first
    """
    column_names = correlation_matrix.columns
    r_values = correlation_matrix.values
    pair_rows = []
    for i in range(len(column_names)):
        for j in range(i + 1, len(column_names)):
            if abs(r_values[i, j]) > limit:
                pair_rows.append({"sensor_a": column_names[i],
                                  "sensor_b": column_names[j],
                                  "r": round(r_values[i, j], 4)})
    strong_pairs_table = pd.DataFrame(pair_rows, columns=["sensor_a", "sensor_b", "r"])
    # Sort by the size of r, ignoring its sign (key=abs), largest first.
    strong_pairs_table = strong_pairs_table.sort_values("r", key=abs, ascending=False)
    return strong_pairs_table.reset_index(drop=True)


def plot_correlation_heatmap(correlation_matrix, title, file_name):
    """
    Draw a correlation matrix as a heatmap and save it. Diverging colours:
    blue = negative, white = 0, red = positive, fixed from -1 to +1 so colours
    mean the same in every heatmap; pairs without an r are light grey.

    Input:  correlation_matrix (DataFrame: square matrix of r values), title (str), file_name (str)
    Output: none (the figure is saved)
    """
    heatmap_colors = plt.get_cmap("RdBu_r").copy()
    heatmap_colors.set_bad("lightgrey")
    figure, axis = plt.subplots(figsize=(9, 8))
    heatmap_image = axis.imshow(correlation_matrix.values, cmap=heatmap_colors, vmin=-1, vmax=1)
    colour_bar = figure.colorbar(heatmap_image, ax=axis, shrink=0.8)
    colour_bar.set_label("Pearson correlation r")
    # Hundreds of sensor names cannot be read, so the tick labels are hidden.
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("Sensor columns (" + str(len(correlation_matrix.columns)) + ", original order)")
    axis.set_ylabel("Sensor columns")
    axis.set_title(title)
    save_figure(figure, file_name)


# ----- 3.1 Structure: shape, data types, duplicated rows (manual 2.2) -----
print("\n--- 3.1 Structure ---")
print("Shape of X (wafers, sensors):", X.shape)
print("Data types of the sensor columns:")
print(X.dtypes.value_counts())

# duplicated() marks a row that is identical to an earlier row in every column
# (pandas treats two missing values in the same place as equal). The check is
# done on the sensors alone and on sensors plus label.
duplicated_rows = X.duplicated().sum()
X_with_label = X.copy()
X_with_label["target"] = y
duplicated_rows_with_label = X_with_label.duplicated().sum()
print("Duplicated rows (sensors only):", duplicated_rows)
print("Duplicated rows (sensors and label):", duplicated_rows_with_label)


# ----- 3.2 Class distribution and imbalance tier (manual 2.3) -----
print("\n--- 3.2 Class distribution ---")
# sort_index() puts pass (0) before fail (1).
class_counts = y.value_counts().sort_index()
class_percent = class_counts / len(y) * 100
class_table = pd.DataFrame({"wafers": class_counts, "percent": class_percent.round(2)})
class_table.index = ["pass (0)", "fail (1)"]
print(class_table)
class_table.to_csv(TABLES_FOLDER / "block3_class_distribution.csv")

# Imbalance tier from the manual (Section 2.3): extreme below 1 %, moderate
# 1 to 20 %, mild 20 to 40 % minority share.
failure_percent = class_percent[1]
if failure_percent < 1:
    imbalance_tier = "extreme"
elif failure_percent <= 20:
    imbalance_tier = "moderate"
elif failure_percent <= 40:
    imbalance_tier = "mild"
else:
    imbalance_tier = "balanced"
print("Failure share:", round(failure_percent, 2), "% -> imbalance tier:", imbalance_tier)

# Bar chart with the count and percentage written on each bar, so the numbers
# can be read without relying on colour.
figure, axis = plt.subplots(figsize=(6, 4))
bar_names = ["Pass (0)", "Fail (1)"]
bar_heights = [class_counts[0], class_counts[1]]
bar_percents = [class_percent[0], class_percent[1]]
axis.bar(bar_names, bar_heights, color=[COLOR_PASS, COLOR_FAIL], width=0.6)
for position in range(len(bar_names)):
    bar_label = str(bar_heights[position]) + " (" + str(round(bar_percents[position], 1)) + " %)"
    axis.text(position, bar_heights[position], bar_label, ha="center", va="bottom")
axis.set_ylabel("Number of wafers")
axis.set_title("Class distribution, full data (" + imbalance_tier + " imbalance)")
axis.spines["top"].set_visible(False)
axis.spines["right"].set_visible(False)
save_figure(figure, "block3_01_class_distribution.png")


# ----- 3.3 Descriptive statistics and scales (manual 2.4.1) -----
print("\n--- 3.3 Descriptive statistics ---")
# describe() gives count (non-missing values), mean, std, min, quartiles and max
# for every column; .T turns it so that each sensor is one row.
feature_summary = X.describe().T
feature_summary.to_csv(TABLES_FOLDER / "block3_descriptive_statistics.csv")
print(feature_summary[["count", "mean", "std", "min", "max"]].head(10))

# The range of means and standard deviations across sensors shows how different
# their scales are.
print("Smallest / largest column mean:", round(feature_summary["mean"].min(), 3), "/",
      round(feature_summary["mean"].max(), 3))
print("Smallest / largest column std: ", round(feature_summary["std"].min(), 3), "/",
      round(feature_summary["std"].max(), 3))
print("Fewest non-missing values in one column:", int(feature_summary["count"].min()), "of", len(X))


# ----- 3.4 Zero-variance columns (manual 4.2, described only) -----
print("\n--- 3.4 Zero-variance columns ---")
# var() is the sample variance of each column (pandas default ddof=1, divides by
# n - 1) and ignores missing values (default skipna=True).
column_variances = X.var()
zero_variance_columns = column_variances[column_variances == 0].index.tolist()
print("Columns with variance exactly 0:", len(zero_variance_columns))


# ----- 3.5 Low-variance analysis: histogram and Pareto chart -----
print("\n--- 3.5 Low-variance analysis ---")
# Histogram and Pareto chart of the column variances (functions above).
plot_variance_histogram(column_variances,
                        "Variance per column, full data (" + str(len(zero_variance_columns))
                        + " zero-variance columns not shown)",
                        "block3_02_variance_histogram.png")
variance_pareto = plot_variance_pareto(column_variances,
                                       "Low-variance Pareto chart, full data (analysis only, no action)",
                                       "block3_03_variance_pareto.png")
variance_pareto.to_csv(TABLES_FOLDER / "block3_variance_pareto.csv", index=False)

# Print the count at each power of ten (every second threshold).
for position in range(0, len(variance_pareto), 2):
    print("  variance <=", variance_pareto["variance_threshold"][position], ":",
          variance_pareto["columns_at_or_below"][position], "columns")


# ----- 3.6 Missing values: totals, per-column chart and Pareto chart (manual 2.5) -----
print("\n--- 3.6 Missing values ---")
total_missing = X.isna().sum().sum()
missing_percent_overall = total_missing / X.size * 100
print("Missing values in total:", total_missing, "of", X.size, "cells (",
      round(missing_percent_overall, 2), "% )")

# isna() is True for a missing value; the mean of True/False per column is the
# share of missing values, times 100 gives the percentage.
missing_percent_per_column = X.isna().mean() * 100
missing_percent_sorted = missing_percent_per_column.sort_values(ascending=False)
print("Columns with at least one missing value:", (missing_percent_per_column > 0).sum())
print("Ten columns with the most missing values (%):")
print(missing_percent_sorted.head(10).round(1))
missing_percent_sorted.round(2).to_csv(TABLES_FOLDER / "block3_missing_per_column.csv",
                                        header=["percent_missing"])

# (a) Per-column chart and (b) Pareto chart (functions above).
plot_missing_per_column(missing_percent_per_column, "Missing values per column, full data",
                        "block3_04_missing_per_column.png")
missing_pareto = plot_missing_pareto(missing_percent_per_column,
                                     "Missingness Pareto chart, full data (no threshold chosen here)",
                                     "block3_05_missing_pareto.png")

for position in range(len(missing_pareto)):
    print("  missing >", missing_pareto["threshold_percent"][position], "% :",
          missing_pareto["columns_removed"][position], "columns")


# ----- 3.7 Missing values by class (manual 2.5) -----
print("\n--- 3.7 Missing values by class ---")
# Number of missing values in each wafer (row), then the average and the median
# per class. groupby(y) splits the wafers into pass (0) and fail (1).
missing_per_wafer = X.isna().sum(axis=1)
missing_by_class_mean = missing_per_wafer.groupby(y).mean()
missing_by_class_median = missing_per_wafer.groupby(y).median()
print("Average missing values per wafer: pass", round(missing_by_class_mean[0], 2),
      "| fail", round(missing_by_class_mean[1], 2))
print("Median missing values per wafer:  pass", missing_by_class_median[0],
      "| fail", missing_by_class_median[1])

figure, axis = plt.subplots(figsize=(6, 4))
class_means = [missing_by_class_mean[0], missing_by_class_mean[1]]
axis.bar(bar_names, class_means, color=[COLOR_PASS, COLOR_FAIL], width=0.6)
for position in range(len(bar_names)):
    axis.text(position, class_means[position], str(round(class_means[position], 1)),
              ha="center", va="bottom")
axis.set_ylabel("Average missing values per wafer")
axis.set_title("Missing values per wafer by class, full data")
axis.spines["top"].set_visible(False)
axis.spines["right"].set_visible(False)
save_figure(figure, "block3_06_missing_by_class.png")


# ----- 3.8 Failures over time (decision D3: parse and explore) -----
print("\n--- 3.8 Failures over time ---")
# to_period("W") assigns each test date to its calendar week (pandas default:
# weeks running Monday to Sunday). Then count wafers and failures per week.
calendar_week = timestamps.dt.to_period("W")
wafers_per_week = y.groupby(calendar_week).size()
failures_per_week = y.groupby(calendar_week).sum()
failure_rate_per_week = failures_per_week / wafers_per_week * 100
weekly_table = pd.DataFrame({"wafers": wafers_per_week,
                             "failures": failures_per_week,
                             "failure_rate_percent": failure_rate_per_week.round(2)})
print(weekly_table)
weekly_table.to_csv(TABLES_FOLDER / "block3_failures_per_week.csv")

# Two charts stacked with one shared date axis, instead of one chart with two
# different y-scales, which is easy to misread. Each week is labelled by its
# Monday, written as "21 Jul".
week_labels = []
for week in weekly_table.index:
    week_labels.append(week.start_time.strftime("%d %b"))

figure, (axis_top, axis_bottom) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
axis_top.bar(week_labels, weekly_table["wafers"], color=COLOR_MAIN)
axis_top.set_ylabel("Wafers tested")
axis_top.set_title("Wafers tested and failure rate per week, Jul to Oct 2008")
axis_bottom.plot(week_labels, weekly_table["failure_rate_percent"], marker="o",
                 linewidth=2, color=COLOR_FAIL)
axis_bottom.axhline(failure_percent, linestyle="--", linewidth=1, color=COLOR_REFERENCE)
axis_bottom.text(0, failure_percent,
                 "overall " + str(round(failure_percent, 1)) + " %",
                 ha="left", va="top", color=COLOR_REFERENCE)
axis_bottom.set_ylabel("Failure rate (%)")
axis_bottom.set_xlabel("Week starting")
axis_bottom.tick_params(axis="x", labelrotation=45)
for axis in [axis_top, axis_bottom]:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
save_figure(figure, "block3_07_failures_over_time.png")


# ----- 3.9 Correlation heatmap (manual 4.6: visualise, do not remove) -----
print("\n--- 3.9 Correlation heatmap ---")
# Pearson correlation between every pair of non-constant columns (a constant
# column has no correlation). pandas' corr() uses, for each pair, the wafers
# where both values exist (pairwise deletion, its default).
non_constant_columns = column_variances[column_variances > 0].index
correlation_matrix = X[non_constant_columns].corr()

# Pairs with |r| > CORRELATION_LIMIT (the papers' removal rule, reported only),
# then the heatmap (functions above).
strong_pairs_full = find_strong_pairs(correlation_matrix, CORRELATION_LIMIT)
print("Non-constant columns in the heatmap:", len(non_constant_columns))
print("Pairs with |r| >", CORRELATION_LIMIT, ":", len(strong_pairs_full))
plot_correlation_heatmap(correlation_matrix,
                         "Correlation heatmap, full data (" + str(len(strong_pairs_full))
                         + " pairs with |r| > " + str(CORRELATION_LIMIT) + ")",
                         "block3_08_correlation_heatmap.png")

# %% Block 4: Stratified train/test split
# Splits the wafers into 80 % training and 20 % test data BEFORE any cleaning,
# so that no rule learned later can be influenced by the test wafers (manual
# Section 3, rule 1). Stratified on the label, because with only 104 failures a
# purely random split could put noticeably more or fewer of them in the test set.

# test_size=TEST_SIZE (0.2) puts 20 % of the wafers in the test set.
# stratify=y draws 20 % from the pass wafers and 20 % from the fail wafers
# separately, so both sets keep the failure share of the full data.
# random_state=SEED fixes the random draw, so the split is identical on every run
# and every laptop. shuffle=True (default) mixes the wafers first, which matters
# because the files are ordered by test date.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
)

# Checks: the two sets together contain every wafer exactly once, and X and y
# still belong together. The index keeps each wafer's original row number, so
# overlap can be checked directly.
print("\n--- 4.1 Split sizes ---")
print("Training set:", X_train.shape, "| test set:", X_test.shape)
wafers_in_both = X_train.index.intersection(X_test.index)
if len(X_train) + len(X_test) != len(X) or len(wafers_in_both) != 0:
    raise ValueError("The split lost, duplicated or shared wafers.")
if not X_train.index.equals(y_train.index) or not X_test.index.equals(y_test.index):
    raise ValueError("X and y rows do not match after the split.")
print("Check passed: every wafer is in exactly one set, X and y match.")

# Processed data: both sets straight after the split. The test set file is only
# a stored copy of the locked test wafers; nothing is learned from it.
save_processed(X_train, y_train, "block4_train_split.csv")
save_processed(X_test, y_test, "block4_test_split.csv")

# Class counts and percentages in the full data, training set and test set
# ("before and after splitting", manual 3.3).
print("\n--- 4.2 Class shares before and after splitting ---")
set_names = ["Full data", "Training set", "Test set"]
set_targets = [y, y_train, y_test]
split_rows = []
for position in range(len(set_names)):
    target = set_targets[position]
    failures = int(target.sum())
    passes = len(target) - failures
    split_rows.append({"set": set_names[position],
                       "wafers": len(target),
                       "pass": passes,
                       "fail": failures,
                       "percent_pass": round(passes / len(target) * 100, 2),
                       "percent_fail": round(failures / len(target) * 100, 2)})
split_table = pd.DataFrame(split_rows)
print(split_table.to_string(index=False))
split_table.to_csv(TABLES_FOLDER / "block4_split_proportions.csv", index=False)

# Bar chart of the failure share in the three sets, each bar labelled with its
# percentage and number of failures.
figure, axis = plt.subplots(figsize=(7, 4))
axis.bar(split_table["set"], split_table["percent_fail"], color=COLOR_FAIL, width=0.6)
for position in range(len(split_table)):
    bar_label = (str(split_table["percent_fail"][position]) + " %\n("
                 + str(split_table["fail"][position]) + " of "
                 + str(split_table["wafers"][position]) + ")")
    axis.text(position, split_table["percent_fail"][position], bar_label,
              ha="center", va="bottom")
axis.set_ylim(0, split_table["percent_fail"].max() * 1.35)
axis.set_ylabel("Failed wafers (%)")
axis.set_title("Failure share before and after the stratified 80/20 split")
axis.spines["top"].set_visible(False)
axis.spines["right"].set_visible(False)
save_figure(figure, "block4_01_failure_share_split.png")

# ---------------------------------------------------------------------------
# THE TEST SET IS NOW LOCKED (manual 3.4, rules 1, 2 and 7).
# From here on:
# - every rule (columns removed, thresholds, outlier bounds, scaling, imputation,
#   selected features, balancing, model, hyperparameters, decision threshold)
#   is learned from X_train / y_train only, and stored in a variable;
# - X_test is only used to APPLY those stored rules, at the final evaluation
#   (Block 17); nothing is ever fitted on it;
# - y_test is not used at all before Block 17.
# ---------------------------------------------------------------------------

# %% Block 5: Rough feature reduction (training set only)
# A first, conservative reduction that removes only columns that clearly carry
# no information or are mostly empty: zero-variance columns, duplicated columns
# and columns above the missing-value threshold (manual Section 4.1 to 4.5;
# "Cleaning of Training Set" in the professor's slides). Every decision is
# calculated on X_train only. Each list of removed columns is stored, so that
# exactly the same columns are removed from X_test at the final evaluation
# (Block 17). X_test is not used in this block.

# Working copy of the training set, so X_train itself stays unchanged.
X_train_reduced = X_train.copy()

# Each step adds one row to this before -> after table.
reduction_rows = [{"step": "Original training set",
                   "columns_before": X_train.shape[1],
                   "columns_removed": 0,
                   "columns_after": X_train.shape[1]}]


# ----- 5.1 Duplicated rows in the training set -----
print("\n--- 5.1 Duplicated rows (training set) ---")
# A duplicated wafer would count twice in training. Rows are only ever removed
# from the training set; test wafers are never removed.
duplicated_training_rows = X_train_reduced.duplicated().sum()
print("Duplicated rows in the training set:", duplicated_training_rows)
if duplicated_training_rows > 0:
    keep_rows = ~X_train_reduced.duplicated()
    X_train_reduced = X_train_reduced[keep_rows]
    y_train = y_train[keep_rows]
    print("Removed duplicated rows; training wafers now:", len(X_train_reduced))


# ----- 5.2 Zero-variance columns (manual 4.2) -----
print("\n--- 5.2 Zero-variance columns (training set) ---")
# Variance per column on the training set (pandas default ddof=1, missing values
# ignored). A column with variance 0 has the same value for every training wafer.
training_variances = X_train_reduced.var()
removed_zero_variance = training_variances[training_variances == 0].index.tolist()
columns_before = X_train_reduced.shape[1]
X_train_reduced = X_train_reduced.drop(columns=removed_zero_variance)
print("Columns before:", columns_before, "| removed:", len(removed_zero_variance),
      "| after:", X_train_reduced.shape[1])
reduction_rows.append({"step": "Zero-variance columns removed",
                       "columns_before": columns_before,
                       "columns_removed": len(removed_zero_variance),
                       "columns_after": X_train_reduced.shape[1]})


# ----- 5.3 Duplicated columns (manual 4.3) -----
print("\n--- 5.3 Duplicated columns (training set) ---")
# .T turns columns into rows, so duplicated() can compare whole columns; it marks
# every column that is identical to an earlier column (the first one is kept).
duplicated_column_mask = X_train_reduced.T.duplicated()
removed_duplicate_columns = X_train_reduced.columns[duplicated_column_mask].tolist()
columns_before = X_train_reduced.shape[1]
X_train_reduced = X_train_reduced.drop(columns=removed_duplicate_columns)
print("Columns before:", columns_before, "| removed:", len(removed_duplicate_columns),
      "| after:", X_train_reduced.shape[1])
reduction_rows.append({"step": "Duplicated columns removed",
                       "columns_before": columns_before,
                       "columns_removed": len(removed_duplicate_columns),
                       "columns_after": X_train_reduced.shape[1]})


# ----- 5.4 Low-variance analysis (analysis only, NO action) -----
print("\n--- 5.4 Low-variance analysis (training set, no action) ---")
# Same histogram and Pareto chart as in Block 3, now on the remaining training
# columns. No column is removed (professor's slide 8): a small variance can
# simply reflect a sensor's unit, not a lack of information.
training_variances = X_train_reduced.var()
plot_variance_histogram(training_variances,
                        "Variance per column, training set (" + str(X_train_reduced.shape[1]) + " columns)",
                        "block5_01_variance_histogram_train.png")
variance_pareto_train = plot_variance_pareto(training_variances,
                                             "Low-variance Pareto chart, training set (analysis only, no action)",
                                             "block5_02_variance_pareto_train.png")
for position in range(0, len(variance_pareto_train), 2):
    print("  variance <=", variance_pareto_train["variance_threshold"][position], ":",
          variance_pareto_train["columns_at_or_below"][position], "columns")


# ----- 5.5 Missing values and threshold (decision D5, manual 4.4 and 4.5) -----
print("\n--- 5.5 Missing values (training set) ---")
missing_percent_train = X_train_reduced.isna().mean() * 100
total_missing_train = X_train_reduced.isna().sum().sum()
print("Missing values in the training set:", total_missing_train, "(",
      round(total_missing_train / X_train_reduced.size * 100, 2), "% of cells )")

plot_missing_per_column(missing_percent_train, "Missing values per column, training set",
                        "block5_03_missing_per_column_train.png")
missing_pareto_train = plot_missing_pareto(missing_percent_train,
                                           "Missingness Pareto chart, training set",
                                           "block5_04_missing_pareto_train.png")
for position in range(len(missing_pareto_train)):
    print("  missing >", missing_pareto_train["threshold_percent"][position], "% :",
          missing_pareto_train["columns_removed"][position], "columns")

# Columns with MORE missing values than MISSING_THRESHOLD are removed (manual 4.5).
removed_missing_columns = missing_percent_train[missing_percent_train > MISSING_THRESHOLD].index.tolist()
print("Columns above", MISSING_THRESHOLD, "% missing (with their % missing):")
print(missing_percent_train[removed_missing_columns].sort_values(ascending=False).round(2))
columns_before = X_train_reduced.shape[1]
X_train_reduced = X_train_reduced.drop(columns=removed_missing_columns)
print("Columns before:", columns_before, "| removed:", len(removed_missing_columns),
      "| after:", X_train_reduced.shape[1])
reduction_rows.append({"step": "Columns with more than " + str(MISSING_THRESHOLD) + " % missing removed",
                       "columns_before": columns_before,
                       "columns_removed": len(removed_missing_columns),
                       "columns_after": X_train_reduced.shape[1]})


# ----- 5.6 Before -> after table and the stored rule for the test set -----
print("\n--- 5.6 Feature reduction summary ---")
reduction_table = pd.DataFrame(reduction_rows)
print(reduction_table.to_string(index=False))
reduction_table.to_csv(TABLES_FOLDER / "block5_feature_reduction.csv", index=False)

# One list of every removed column with its reason. This is the stored rule:
# at the final evaluation (Block 17), X_test.drop(columns=columns_removed_in_block5)
# removes exactly these columns from the test set.
columns_removed_in_block5 = removed_zero_variance + removed_duplicate_columns + removed_missing_columns
removed_reasons = []
for column in removed_zero_variance:
    removed_reasons.append({"column": column, "reason": "zero variance"})
for column in removed_duplicate_columns:
    removed_reasons.append({"column": column, "reason": "duplicate of another column"})
for column in removed_missing_columns:
    removed_reasons.append({"column": column,
                            "reason": "more than " + str(MISSING_THRESHOLD) + " % missing"})
pd.DataFrame(removed_reasons).to_csv(TABLES_FOLDER / "block5_removed_columns.csv", index=False)
print("Columns removed in total:", len(columns_removed_in_block5),
      "| remaining training columns:", X_train_reduced.shape[1])

# Processed data: the training set after the feature reduction (450 columns).
save_processed(X_train_reduced, y_train, "block5_train_reduced.csv")

# %% Block 6: Correlation analysis on the training set (visualise only)
# Shows how strongly the 450 remaining training columns move together (manual
# Section 4.6). Unlike Munirathinam and Ramadoss (2016), who dropped one sensor
# of every pair with |r| >= 0.8, NO column is removed here: without a measure of
# which sensor predicts failure better, the choice would be arbitrary. Redundancy
# is handled later by Boruta (Block 10), which uses the label.
print("\n--- 6.1 Correlation analysis (training set) ---")

# Pearson correlation between every pair of remaining training columns. pandas'
# corr() uses method="pearson" by default and, for each pair, only the wafers
# where both values exist (pairwise deletion).
correlation_matrix_train = X_train_reduced.corr()

# Pairs above the limit, strongest first; saved as a table, nothing removed.
strong_pairs_train = find_strong_pairs(correlation_matrix_train, CORRELATION_LIMIT)
strong_pairs_train.to_csv(TABLES_FOLDER / "block6_strong_correlation_pairs.csv", index=False)
total_pairs = len(correlation_matrix_train.columns) * (len(correlation_matrix_train.columns) - 1) // 2
print("Columns:", len(correlation_matrix_train.columns), "| pairs in total:", total_pairs)
print("Pairs with |r| >", CORRELATION_LIMIT, ":", len(strong_pairs_train),
      "(", round(len(strong_pairs_train) / total_pairs * 100, 2), "% of all pairs )")

# How many different columns appear in at least one strong pair.
columns_in_strong_pairs = set(strong_pairs_train["sensor_a"]).union(set(strong_pairs_train["sensor_b"]))
print("Columns involved in at least one strong pair:", len(columns_in_strong_pairs))
print("Ten strongest pairs:")
print(strong_pairs_train.head(10).to_string(index=False))

plot_correlation_heatmap(correlation_matrix_train,
                         "Correlation heatmap, training set (" + str(len(strong_pairs_train))
                         + " pairs with |r| > " + str(CORRELATION_LIMIT) + ", none removed)",
                         "block6_01_correlation_heatmap_train.png")
print("No columns removed; training columns remain:", X_train_reduced.shape[1])

# %% Block 7: Outliers with the 3 sigma rule (training set only)
# Flags extreme values per sensor with the 3 sigma rule and applies the chosen
# treatment (decision D7) to the training set (manual Section 5; rule 4: no IQR).
# The mean, standard deviation and bounds are computed on the training set only
# and stored, so the same bounds can be applied to the test set in Block 17.
# All three treatments are written as functions so the grid search (Block 15)
# can compare them.


def find_outliers(data, lower_bounds, upper_bounds):
    """
    Mark every value below its column's lower bound or above its upper bound.
    Missing values are never marked (a comparison with NaN is always False).

    Input:  data (DataFrame), lower_bounds and upper_bounds (Series: one bound per column)
    Output: DataFrame of True/False with the same shape as data (True = outlier)
    """
    # axis="columns" lines each bound up with the column of the same name.
    below = data.lt(lower_bounds, axis="columns")
    above = data.gt(upper_bounds, axis="columns")
    return below | above


def treat_outliers_nothing(data, lower_bounds, upper_bounds):
    """
    Option "do nothing": keep every value as measured (the baseline).

    Input:  data (DataFrame), lower_bounds and upper_bounds (Series; not used, kept so
            all three options are called the same way)
    Output: an unchanged copy of data
    """
    return data.copy()


def treat_outliers_cap(data, lower_bounds, upper_bounds):
    """
    Option "cap" (winsorising): pull every value beyond a bound back to that bound.

    Input:  data (DataFrame), lower_bounds and upper_bounds (Series: one bound per column)
    Output: a copy of data with outliers replaced by the nearest bound
    """
    # clip() with axis=1 uses each column's own bounds; missing values stay missing.
    return data.clip(lower=lower_bounds, upper=upper_bounds, axis=1)


def treat_outliers_remove(data, lower_bounds, upper_bounds):
    """
    Option "remove": replace every outlier with a missing value (a gap that the
    kNN imputation in Block 8 has to fill).

    Input:  data (DataFrame), lower_bounds and upper_bounds (Series: one bound per column)
    Output: a copy of data with outliers replaced by NaN
    """
    # mask() puts NaN wherever the outlier table is True.
    return data.mask(find_outliers(data, lower_bounds, upper_bounds))


def apply_outlier_treatment(option, data, lower_bounds, upper_bounds):
    """
    Apply one of the three outlier treatments by name.

    Input:  option (str: "nothing", "cap" or "remove"), data (DataFrame),
            lower_bounds and upper_bounds (Series: one bound per column)
    Output: the treated copy of data
    """
    if option == "nothing":
        return treat_outliers_nothing(data, lower_bounds, upper_bounds)
    elif option == "cap":
        return treat_outliers_cap(data, lower_bounds, upper_bounds)
    elif option == "remove":
        return treat_outliers_remove(data, lower_bounds, upper_bounds)
    else:
        raise ValueError("Unknown outlier treatment: " + str(option))


# ----- 7.1 Bounds from the training set (stored for the test set) -----
print("\n--- 7.1 Outlier bounds (training set) ---")
# mean() and std() per column on the training set only. std() is the sample
# standard deviation (pandas default ddof=1); both ignore missing values.
training_means = X_train_reduced.mean()
training_stds = X_train_reduced.std()
lower_bounds = training_means - OUTLIER_SIGMA * training_stds
upper_bounds = training_means + OUTLIER_SIGMA * training_stds
bounds_table = pd.DataFrame({"mean": training_means, "std": training_stds,
                             "lower_bound": lower_bounds, "upper_bound": upper_bounds})
bounds_table.to_csv(TABLES_FOLDER / "block7_outlier_bounds.csv")
print("Bounds stored for", len(bounds_table), "columns (mean +/-", OUTLIER_SIGMA, "standard deviations)")


# ----- 7.2 How many values are flagged? -----
print("\n--- 7.2 Flagged values (training set) ---")
outlier_mask = find_outliers(X_train_reduced, lower_bounds, upper_bounds)
flags_per_column = outlier_mask.sum()
total_flagged = flags_per_column.sum()
measured_values = X_train_reduced.notna().sum().sum()
print("Values flagged:", total_flagged, "of", measured_values, "measured values (",
      round(total_flagged / measured_values * 100, 2), "% )")
# Under a perfect bell curve about 0.27 % of values lie beyond 3 sigma.
print("Expected under a normal distribution (0.27 %):", round(measured_values * 0.0027))
print("Sensors with at least one flagged value:", (flags_per_column > 0).sum(), "of",
      X_train_reduced.shape[1])

top10_outlier_sensors = flags_per_column.sort_values(ascending=False).head(10)
print("Ten most affected sensors (flagged values):")
print(top10_outlier_sensors)
top10_outlier_sensors.to_csv(TABLES_FOLDER / "block7_top10_outlier_sensors.csv",
                             header=["flagged_values"])

# Are outliers more common in failed wafers? Count flags per wafer, then average
# per class (training labels only, describing, nothing is fitted).
flags_per_wafer = outlier_mask.sum(axis=1)
flags_by_class = flags_per_wafer.groupby(y_train).mean()
print("Average flagged values per wafer: pass", round(flags_by_class[0], 2),
      "| fail", round(flags_by_class[1], 2))


# ----- 7.3 Compare the three options, then apply the chosen one (D7) -----
print("\n--- 7.3 Outlier treatment options (training set) ---")
missing_before_treatment = X_train_reduced.isna().sum().sum()
option_rows = []
for option in ["nothing", "cap", "remove"]:
    treated = apply_outlier_treatment(option, X_train_reduced, lower_bounds, upper_bounds)
    # A value counts as changed if it differs from before; NaN != NaN is True in
    # pandas, so cells missing before AND after are excluded.
    both_missing = X_train_reduced.isna() & treated.isna()
    values_changed = ((treated != X_train_reduced) & ~both_missing).sum().sum()
    option_rows.append({"option": option,
                        "values_changed": values_changed,
                        "missing_before": missing_before_treatment,
                        "missing_after": treated.isna().sum().sum()})
options_table = pd.DataFrame(option_rows)
print(options_table.to_string(index=False))
options_table.to_csv(TABLES_FOLDER / "block7_outlier_options.csv", index=False)

# Apply the chosen treatment to the training set. The stored bounds are applied
# to the test set with the same function in Block 17.
X_train_treated = apply_outlier_treatment(OUTLIER_TREATMENT, X_train_reduced, lower_bounds, upper_bounds)
print("Chosen treatment:", OUTLIER_TREATMENT, "| training shape:", X_train_treated.shape,
      "| missing values for Phase 2 to impute:", X_train_treated.isna().sum().sum())

# Processed data: the training set after outlier capping, the input for Phase 2.
save_processed(X_train_treated, y_train, "block7_train_capped.csv")


# ----- 7.4 Before/after boxplot of the most affected sensor -----
# The boxplot is only a picture: its whiskers follow the usual 1.5 x IQR drawing
# convention, while outliers were detected with the 3 sigma bounds (dashed lines).
example_sensor = flags_per_column.idxmax()
values_before = X_train_reduced[example_sensor].dropna()
values_after = X_train_treated[example_sensor].dropna()
figure, axis = plt.subplots(figsize=(7, 4))
axis.boxplot([values_before, values_after],
             tick_labels=["Before", "After (" + OUTLIER_TREATMENT + ")"])
axis.axhline(lower_bounds[example_sensor], linestyle="--", linewidth=1, color=COLOR_REFERENCE)
axis.axhline(upper_bounds[example_sensor], linestyle="--", linewidth=1, color=COLOR_REFERENCE)
axis.text(2.45, upper_bounds[example_sensor], "+3 sigma", va="bottom", ha="right", color=COLOR_REFERENCE)
axis.text(2.45, lower_bounds[example_sensor], "-3 sigma", va="top", ha="right", color=COLOR_REFERENCE)
axis.set_ylabel(example_sensor + " value")
axis.set_title(example_sensor + ": " + str(flags_per_column[example_sensor])
               + " flagged values, before and after treatment (training set)")
axis.spines["top"].set_visible(False)
axis.spines["right"].set_visible(False)
save_figure(figure, "block7_01_outlier_boxplot_" + example_sensor + ".png")

print("\nPhase 1 steps finished; Phase 2 starts.")


# %% Block 8: Missing-value check, scaling and kNN imputation (training set only)
# Phase 2 starts from the capped training set of Block 7. Before any value is
# filled in, every gap must be a real missing value (NaN), because the imputer
# only recognises NaN as missing (manual week 3; rule 5). Scaling and kNN
# imputation follow in this block (Step 2.2). The test set is not used.

# ----- 8.1 Missing values before imputation -----
print("\n--- 8.1 Missing values before imputation (training set) ---")

# Every column must be numeric: a text column would mean that something other
# than "NaN" marked a gap, and the imputer could not handle it.
non_numeric_columns = []
for column in X_train_treated.columns:
    if not pd.api.types.is_numeric_dtype(X_train_treated[column]):
        non_numeric_columns.append(column)
print("Non-numeric columns:", len(non_numeric_columns))

# Infinite values (inf) are not counted as missing by isna() and would break the
# distance calculation of kNN. np.isinf() returns False for NaN, so only real
# infinities are counted.
infinite_values = np.isinf(X_train_treated.to_numpy()).sum()
print("Infinite values:", infinite_values)
if len(non_numeric_columns) > 0 or infinite_values > 0:
    raise ValueError("Some gaps are not flagged as NaN; check the data before imputation.")

# The starting point for imputation: every gap is now a NaN. Capping (D7)
# created no new gaps, so this equals the count after Block 5.
missing_before_imputation = X_train_treated.isna().sum().sum()
print("Missing values to impute:", missing_before_imputation, "of", X_train_treated.size,
      "cells (", round(missing_before_imputation / X_train_treated.size * 100, 2), "% )")
missing_percent_before = X_train_treated.isna().mean() * 100
print("Columns with at least one missing value:", (missing_percent_before > 0).sum(),
      "of", X_train_treated.shape[1])
print("Ten columns with the most missing values (%):")
print(missing_percent_before.sort_values(ascending=False).head(10).round(2))

# ----- 8.2 Min-max scaling (decision D8, fitted on the training set only) -----
print("\n--- 8.2 Min-max scaling (training set) ---")
# kNN imputation measures distances across all sensors, so every sensor must be
# on the same scale; otherwise sensors measured in thousands would decide alone
# which wafers count as neighbours. MinMaxScaler maps each column to 0 to 1
# (default feature_range=(0, 1)): (value - minimum) / (maximum - minimum).
# fit() learns each column's minimum and maximum from the training set only;
# missing values are ignored when learning and stay missing after transform().
scaler = MinMaxScaler()
scaler.fit(X_train_treated)
X_train_scaled = pd.DataFrame(scaler.transform(X_train_treated),
                              columns=X_train_treated.columns, index=X_train_treated.index)
print("Scaled training values from", round(X_train_scaled.min().min(), 4), "to",
      round(X_train_scaled.max().max(), 4), "| missing values still:", X_train_scaled.isna().sum().sum())


# ----- 8.3 kNN imputation (decision D9, fitted on the training set only) -----
print("\n--- 8.3 kNN imputation (training set) ---")
# For each gap, KNNImputer finds the KNN_NEIGHBORS most similar training wafers
# and fills the gap with their average value for that sensor. Defaults used:
# weights="uniform" (all neighbours count equally) and metric="nan_euclidean"
# (the distance between two wafers uses only the sensors both have measured).
# fit() stores the scaled training wafers as the pool of possible neighbours.
imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS)
imputer.fit(X_train_scaled)
X_train_imputed = pd.DataFrame(imputer.transform(X_train_scaled),
                               columns=X_train_scaled.columns, index=X_train_scaled.index)

# Checks: no gap may remain, and every value that was measured must be unchanged
# (only the gaps are filled). 1e-9 allows for tiny rounding differences.
missing_after_imputation = X_train_imputed.isna().sum().sum()
measured = X_train_scaled.notna()
largest_change = (X_train_imputed[measured] - X_train_scaled[measured]).abs().max().max()
print("Missing values: before", missing_before_imputation, "| after", missing_after_imputation)
print("Largest change to a measured value:", largest_change)
if missing_after_imputation != 0 or largest_change > 1e-9:
    raise ValueError("Imputation check failed: gaps remain or measured values changed.")

# The fitted scaler and imputer are the stored rules for the final evaluation
# (Phase 3, Block 17): the test set will only get scaler.transform() and then
# imputer.transform(); nothing is ever fitted on it. The test set is not used here.

# Summary table and processed data (scaled 0 to 1, no gaps).
imputation_summary = pd.DataFrame([{"scaling": "min-max (0 to 1), fitted on training set",
                                    "imputation": "KNNImputer, k = " + str(KNN_NEIGHBORS) + ", uniform weights, nan_euclidean",
                                    "training_wafers": X_train_imputed.shape[0],
                                    "columns": X_train_imputed.shape[1],
                                    "missing_before": missing_before_imputation,
                                    "missing_after": missing_after_imputation}])
imputation_summary.to_csv(TABLES_FOLDER / "block8_imputation_summary.csv", index=False)
save_processed(X_train_imputed, y_train, "block8_train_imputed.csv")


# ----- 8.4 Before/after histogram for the most incomplete sensor -----
# The sensor with the most gaps shows the effect of imputation most clearly. The
# imputed values are turned back into the sensor's original units with
# inverse_transform(), only for this picture, so the axis is readable.
example_sensor = missing_percent_before.idxmax()
values_observed = X_train_treated[example_sensor].dropna()
X_train_imputed_original_units = pd.DataFrame(scaler.inverse_transform(X_train_imputed),
                                              columns=X_train_imputed.columns,
                                              index=X_train_imputed.index)
values_after = X_train_imputed_original_units[example_sensor]
print("\n" + example_sensor + ":", round(missing_percent_before[example_sensor], 2), "% missing")
print("  observed only:       n =", len(values_observed), "| mean", round(values_observed.mean(), 3),
      "| std", round(values_observed.std(), 3))
print("  observed + imputed:  n =", len(values_after), "| mean", round(values_after.mean(), 3),
      "| std", round(values_after.std(), 3))

# Same bins in both panels (np.linspace: 31 edges = 30 equal bins over the
# observed range), so the two histograms can be compared directly.
bin_edges = np.linspace(values_after.min(), values_after.max(), 31)
figure, (axis_left, axis_right) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
axis_left.hist(values_observed, bins=bin_edges, color=COLOR_MAIN, edgecolor="white")
axis_left.set_title("Before: observed only (n = " + str(len(values_observed)) + ")")
axis_right.hist(values_after, bins=bin_edges, color=COLOR_MAIN, edgecolor="white")
axis_right.set_title("After: observed + imputed (n = " + str(len(values_after)) + ")")
axis_left.set_ylabel("Number of training wafers")
for axis in [axis_left, axis_right]:
    axis.set_xlabel(example_sensor + " value (original units)")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
figure.suptitle(example_sensor + " (" + str(round(missing_percent_before[example_sensor], 1))
                + " % missing): kNN imputation, k = " + str(KNN_NEIGHBORS))
save_figure(figure, "block8_01_imputation_check_" + example_sensor + ".png")


# %% Block 9: First random forest - training fit vs. cross-validation
# Trains a first random forest on the imputed training data (manual week 3,
# step 5; Section 11.2) and evaluates it twice: on the same wafers it learned
# from (training fit) and with 5-fold stratified cross-validation, where every
# wafer is predicted by a model that never saw it. Only the training set is
# used; the test set stays locked.


def evaluate_predictions(y_true, y_pred):
    """
    Compute the confusion matrix, the usual metrics and the total cost.

    Input:  y_true (Series or array of true labels, 1 = fail), y_pred (predicted labels)
    Output: dictionary with TN, FP, FN, TP, accuracy, recall, precision,
            specificity, F1 and total cost in euros
    """
    # labels=[0, 1] fixes the order (pass, fail) and keeps a 2 x 2 matrix even
    # if one class is never predicted. ravel() reads it as TN, FP, FN, TP.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = tn + fp + fn + tp
    # Each ratio is set to 0 when its denominator is 0 (for example precision
    # when no wafer is predicted as failed), instead of dividing by zero.
    accuracy = (tp + tn) / total
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    cost = COST_FALSE_POSITIVE * fp + COST_FALSE_NEGATIVE * fn
    # int() and float() turn numpy numbers into plain Python numbers, so they print cleanly.
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": round(float(accuracy), 4), "recall": round(float(recall), 4),
            "precision": round(float(precision), 4), "specificity": round(float(specificity), 4),
            "F1": round(float(f1), 4), "cost_eur": int(cost)}


# ----- 9.1 Training fit: the model is scored on the wafers it learned from -----
print("\n--- 9.1 Random forest, training fit ---")
# Defaults used (scikit-learn 1.9.1): n_estimators=100 trees; max_features="sqrt"
# (each split chooses among about 21 of the 450 sensors); bootstrap=True (each
# tree learns from a random sample of wafers drawn with replacement); no depth
# limit; no class weights. random_state=SEED makes the forest identical on every run.
random_forest = RandomForestClassifier(random_state=SEED)
random_forest.fit(X_train_imputed, y_train)
# predict() calls a wafer "fail" when more than half of the trees vote fail (threshold 0.5).
training_fit_results = evaluate_predictions(y_train, random_forest.predict(X_train_imputed))
print(training_fit_results)


# ----- 9.2 5-fold stratified cross-validation (a visible loop inside a function) -----
print("\n--- 9.2 Random forest, " + str(CV_FOLDS) + "-fold stratified cross-validation ---")
# StratifiedKFold cuts the training wafers into CV_FOLDS parts with the same
# failure share. shuffle=True mixes the wafers first (they are in time order);
# random_state=SEED makes the parts identical on every run, so every model in
# this script is evaluated on exactly the same five parts.
cv_splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)


def balance_training_data(option, X_data, y_data):
    """
    Balance training data with one of the options screened in Block 11.
    Only ever called on training data (inside a training fold or on the
    training set), never on data that is evaluated.

    Input:  option (str: "none", "class_weight", "SMOTE", "ROSE" or "ADASYN"),
            X_data (DataFrame of training features), y_data (Series of training labels)
    Output: X_balanced (DataFrame), y_balanced (Series), class_weight (None or
            "balanced", the setting for the random forest)
    """
    # "none" and "class_weight" create no new wafers; class weights are handled by the model.
    if option == "none":
        return X_data, y_data, None
    if option == "class_weight":
        return X_data, y_data, "balanced"
    # sampling_strategy="auto" (default) adds failed wafers until both classes
    # are equally large; random_state=SEED makes the new wafers repeatable.
    if option == "SMOTE":
        # k_neighbors=5 (default): a new wafer lies between a failed wafer and
        # one of its 5 nearest failed neighbours.
        sampler = SMOTE(random_state=SEED)
    elif option == "ROSE":
        # A smoothed bootstrap: copy a random failed wafer and add a little
        # random noise; shrinkage sets how much (1.0 = ROSE's default rule).
        sampler = RandomOverSampler(shrinkage=ROSE_SHRINKAGE, random_state=SEED)
    elif option == "ADASYN":
        # n_neighbors=5 (default): like SMOTE, but more new wafers where failed
        # wafers are surrounded by passed ones.
        sampler = ADASYN(random_state=SEED)
    else:
        raise ValueError("Unknown balancing option: " + str(option))
    X_balanced, y_balanced = sampler.fit_resample(X_data, y_data)
    # The original wafers come first, the new ones are appended; the original
    # wafer numbers are kept and the new rows are named synthetic_1, synthetic_2, ...
    new_index = list(X_data.index)
    for number in range(1, len(X_balanced) - len(X_data) + 1):
        new_index.append("synthetic_" + str(number))
    X_balanced.index = new_index
    y_balanced = pd.Series(np.asarray(y_balanced), index=new_index, name=y_data.name)
    return X_balanced, y_balanced, None


def cross_validate_random_forest(X_data, y_data, balancing="none"):
    """
    Evaluate a random forest (seed SEED, default settings) with the stratified
    cross-validation parts of cv_splitter: for each part, a new forest learns
    from the other parts and predicts this one, so every wafer is predicted
    exactly once by a model that never saw it. If a balancing option is given,
    only the training part of each fold is balanced; the part that is
    evaluated keeps its real class mix.

    Input:  X_data (DataFrame of training features), y_data (Series of training labels),
            balancing (str, default "none": see balance_training_data)
    Output: fold_table (DataFrame: one row of results per part),
            cv_predictions (Series: out-of-fold predicted class per wafer),
            cv_probabilities (Series: out-of-fold predicted failure probability per wafer)
    """
    cv_predictions = pd.Series(0, index=y_data.index)
    cv_probabilities = pd.Series(0.0, index=y_data.index)
    fold_rows = []
    fold_number = 0
    for train_positions, validation_positions in cv_splitter.split(X_data, y_data):
        fold_number = fold_number + 1
        # iloc selects rows by position: 4 parts to learn from, 1 part to evaluate on.
        X_fold_train = X_data.iloc[train_positions]
        y_fold_train = y_data.iloc[train_positions]
        X_fold_validation = X_data.iloc[validation_positions]
        y_fold_validation = y_data.iloc[validation_positions]

        # Balance the training part only (rule 8); the validation part is left as it is.
        X_fold_balanced, y_fold_balanced, class_weight = balance_training_data(balancing, X_fold_train, y_fold_train)

        # A new forest with the same settings for every part, trained without the validation part.
        fold_model = RandomForestClassifier(random_state=SEED, class_weight=class_weight)
        fold_model.fit(X_fold_balanced, y_fold_balanced)
        fold_predictions = fold_model.predict(X_fold_validation)
        # predict_proba() gives [P(pass), P(fail)] per wafer; column 1 is the share of trees voting fail.
        fold_probabilities = fold_model.predict_proba(X_fold_validation)[:, 1]

        cv_predictions.loc[X_fold_validation.index] = fold_predictions
        cv_probabilities.loc[X_fold_validation.index] = fold_probabilities
        fold_result = evaluate_predictions(y_fold_validation, fold_predictions)
        fold_result["fold"] = fold_number
        fold_result["validation_wafers"] = len(y_fold_validation)
        fold_result["failed_wafers"] = int(y_fold_validation.sum())
        fold_rows.append(fold_result)

    fold_table = pd.DataFrame(fold_rows)
    fold_table = fold_table[["fold", "validation_wafers", "failed_wafers", "TN", "FP", "FN", "TP",
                             "accuracy", "recall", "precision", "cost_eur"]]
    return fold_table, cv_predictions, cv_probabilities


fold_table, cv_predictions, cv_probabilities = cross_validate_random_forest(X_train_imputed, y_train)
print(fold_table.to_string(index=False))
fold_table.to_csv(TABLES_FOLDER / "block9_cv_folds.csv", index=False)

# All five validation parts together: every training wafer predicted once by a model that never saw it.
cross_validation_results = evaluate_predictions(y_train, cv_predictions)
print("All folds together:", cross_validation_results)
print("Highest predicted failure probability:", round(cv_probabilities.max(), 3),
      "| wafers with probability >= 0.5:", (cv_probabilities >= 0.5).sum(),
      "| >= 0.3:", (cv_probabilities >= 0.3).sum())


# ----- 9.3 Side by side: training fit vs. cross-validation -----
comparison_table = pd.DataFrame([training_fit_results, cross_validation_results],
                                index=["training fit (seen wafers)",
                                       str(CV_FOLDS) + "-fold CV (unseen wafers)"])
print("\n" + comparison_table.to_string())
comparison_table.to_csv(TABLES_FOLDER / "block9_train_vs_cv.csv", index_label="evaluation")

# Two confusion matrices side by side. Rows = true class, columns = predicted
# class; each cell shows its count. One blue scale for both (vmin/vmax fixed).
figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
panel_results = [training_fit_results, cross_validation_results]
panel_titles = ["Training fit (wafers the model has seen)",
                str(CV_FOLDS) + "-fold CV (wafers the model has not seen)"]
for position in range(2):
    result = panel_results[position]
    matrix = np.array([[result["TN"], result["FP"]], [result["FN"], result["TP"]]])
    axis = axes[position]
    axis.imshow(matrix, cmap="Blues", vmin=0, vmax=len(y_train))
    cell_labels = [["TN", "FP"], ["FN", "TP"]]
    for row in range(2):
        for column in range(2):
            count = matrix[row, column]
            # Dark cells get white text, light cells dark text, so every number stays readable.
            text_color = "white" if count > len(y_train) / 2 else "#222222"
            axis.text(column, row, cell_labels[row][column] + "\n" + str(count),
                      ha="center", va="center", fontsize=13, color=text_color)
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["Pass (0)", "Fail (1)"])
    axis.set_yticks([0, 1])
    axis.set_yticklabels(["Pass (0)", "Fail (1)"])
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(panel_titles[position] + "\nrecall " + str(result["recall"])
                   + " | accuracy " + str(result["accuracy"]) + " | cost EUR " + format(result["cost_eur"], ","),
                   fontsize=10)
figure.suptitle("First random forest on the training set (" + str(len(y_train)) + " wafers, "
                + str(int(y_train.sum())) + " failed)")
# Extra space between the panels and below the main title so no labels overlap.
figure.subplots_adjust(top=0.80, wspace=0.45)
save_figure(figure, "block9_01_confusion_matrices.png")

# %% Block 10: Boruta feature selection (training set only)
# Keeps only the sensors that predict failure better than chance (manual week 4;
# Section 7). Boruta compares every sensor with shuffled "shadow" copies over
# many random-forest rounds and confirms, rejects or leaves it tentative.
# It uses the label, so it is fitted on X_train_imputed / y_train only; the
# list of selected sensors is the stored rule applied to the test set in Block 17.

# ----- 10.1 Run Boruta -----
print("\n--- 10.1 Boruta feature selection (training set) ---")
# The forest inside Boruta: class_weight="balanced" weights each failed wafer by
# about 1,170 / 83 so the few failures count as much as the many passes;
# max_depth=5 keeps the trees shallow, as the Boruta documentation recommends,
# which gives steadier importances; n_jobs=None (default) runs on one processor
# thread: about 25 s slower than all cores, but the same result and no
# multi-threading warnings (DEBUG_LOG.md, 20 Oct); random_state=SEED makes it repeatable.
boruta_forest = RandomForestClassifier(class_weight="balanced", max_depth=5,
                                       n_jobs=None, random_state=SEED)
# n_estimators="auto" lets Boruta choose the number of trees from the number of
# columns. Defaults kept: perc=100 (a sensor must beat the BEST shadow), alpha=0.05
# (significance level), two_step=True (test corrected for testing 450 sensors).
boruta_selector = BorutaPy(boruta_forest, n_estimators="auto", max_iter=BORUTA_MAX_ITER,
                           random_state=SEED, verbose=0)
# BorutaPy expects plain arrays, so .to_numpy() is passed; fitted on training data only.
boruta_selector.fit(X_train_imputed.to_numpy(), y_train.to_numpy())

# support_ marks confirmed sensors, support_weak_ tentative ones; all others are rejected.
confirmed_features = X_train_imputed.columns[boruta_selector.support_].tolist()
tentative_features = X_train_imputed.columns[boruta_selector.support_weak_].tolist()
rejected_count = X_train_imputed.shape[1] - len(confirmed_features) - len(tentative_features)
print("Confirmed:", len(confirmed_features), confirmed_features)
print("Tentative:", len(tentative_features), tentative_features)
print("Rejected: ", rejected_count)

# Decision D10: the stored rule for the test set is this list of sensor names.
selected_features = list(confirmed_features)
if KEEP_TENTATIVE:
    selected_features = selected_features + tentative_features
print("Selected (KEEP_TENTATIVE =", str(KEEP_TENTATIVE) + "):", len(selected_features), "features")


# ----- 10.2 Ranking table and chart -----
# importance_history_ holds each sensor's importance in every round (missing once
# a sensor is rejected); the median over the rounds summarises it robustly.
median_importance = pd.Series(np.nanmedian(boruta_selector.importance_history_, axis=0),
                              index=X_train_imputed.columns)
status = pd.Series("rejected", index=X_train_imputed.columns)
status[confirmed_features] = "confirmed"
status[tentative_features] = "tentative"
ranking_table = pd.DataFrame({"boruta_rank": boruta_selector.ranking_,
                              "status": status,
                              "median_importance": median_importance.round(6)})
ranking_table = ranking_table.sort_values(["boruta_rank", "median_importance"], ascending=[True, False])
ranking_table.to_csv(TABLES_FOLDER / "block10_boruta_ranking.csv", index_label="sensor")

# Horizontal bar chart of the 30 highest-ranked sensors, coloured by status
# (green confirmed, yellow tentative, grey rejected), most important at the top.
top_ranked = ranking_table.head(30).iloc[::-1]
status_colors = {"confirmed": "#008300", "tentative": "#eda100", "rejected": "#a6a6a6"}
bar_colors = []
for sensor_status in top_ranked["status"]:
    bar_colors.append(status_colors[sensor_status])
figure, axis = plt.subplots(figsize=(8, 8))
axis.barh(top_ranked.index, top_ranked["median_importance"], color=bar_colors)
# Legend: one coloured square (Patch) per status.
legend_handles = []
for sensor_status in ["confirmed", "tentative", "rejected"]:
    legend_handles.append(Patch(color=status_colors[sensor_status], label=sensor_status))
axis.legend(handles=legend_handles, loc="lower right")
axis.set_xlabel("Median importance over Boruta rounds")
axis.set_title("Boruta ranking, training set: " + str(len(confirmed_features)) + " confirmed, "
               + str(len(tentative_features)) + " tentative, " + str(rejected_count) + " rejected\n"
               + "(30 highest-ranked of " + str(X_train_imputed.shape[1]) + " sensors)")
axis.spines["top"].set_visible(False)
axis.spines["right"].set_visible(False)
save_figure(figure, "block10_01_boruta_ranking.png")


# ----- 10.3 Random forest with the selected sensors vs. Block 9 -----
print("\n--- 10.3 Cross-validated random forest: all vs. selected sensors ---")
# The same function, the same five parts and the same forest settings as in
# Block 9, so the only difference between the rows is the set of sensors.
feature_sets = {"all sensors (Block 9)": list(X_train_imputed.columns),
                "Boruta confirmed only": confirmed_features,
                "Boruta selected (D10)": selected_features}
comparison_rows = []
for set_name in feature_sets:
    columns = feature_sets[set_name]
    folds, predictions, probabilities = cross_validate_random_forest(X_train_imputed[columns], y_train)
    row = evaluate_predictions(y_train, predictions)
    row["feature_set"] = set_name
    row["features"] = len(columns)
    row["highest_fail_probability"] = round(float(probabilities.max()), 3)
    row["wafers_at_or_above_0.3"] = int((probabilities >= 0.3).sum())
    comparison_rows.append(row)
selection_comparison = pd.DataFrame(comparison_rows)
selection_comparison = selection_comparison[["feature_set", "features", "TN", "FP", "FN", "TP",
                                             "accuracy", "recall", "precision", "cost_eur",
                                             "highest_fail_probability", "wafers_at_or_above_0.3"]]
print(selection_comparison.to_string(index=False))
selection_comparison.to_csv(TABLES_FOLDER / "block10_cv_comparison.csv", index=False)

# Processed data: the training set with only the selected sensors (scaled, imputed).
X_train_selected = X_train_imputed[selected_features]
save_processed(X_train_selected, y_train, "block10_train_selected.csv")

# %% Block 11: Class imbalance - screening five balancing options (training set only)
# With 1,170 passed and 83 failed training wafers, a model can be right 93 % of
# the time by ignoring failures. Balancing changes what the model learns from
# (manual Section 9). Five options are screened on the random forest with the
# 18 Boruta sensors and the same five CV parts, balancing only the training
# part of each fold (rule 8). ADASYN may be dropped only by the drop rule fixed
# in advance (DECISIONS.md, 20 Oct); the working choice is decision D11.
# The test set is never balanced and is not used here.

# ----- 11.1 Class counts before and after balancing (whole training set, for the chart) -----
print("\n--- 11.1 Class counts before and after balancing (training set) ---")
count_rows = [{"version": "original", "pass": int((y_train == 0).sum()), "fail": int((y_train == 1).sum())}]
for option in ["SMOTE", "ROSE", "ADASYN"]:
    X_balanced, y_balanced, unused_weight = balance_training_data(option, X_train_selected, y_train)
    count_rows.append({"version": option, "pass": int((y_balanced == 0).sum()), "fail": int((y_balanced == 1).sum())})
class_counts_table = pd.DataFrame(count_rows)
print(class_counts_table.to_string(index=False))
print("class_weight: no new wafers; each failed wafer counts about",
      round((y_train == 0).sum() / (y_train == 1).sum(), 1), "times as much in training")

# Grouped bars: pass and fail count per version, each bar labelled with its number.
figure, axis = plt.subplots(figsize=(8, 4.5))
positions = np.arange(len(class_counts_table))
bar_width = 0.38
axis.bar(positions - bar_width / 2, class_counts_table["pass"], bar_width, color=COLOR_PASS, label="Pass (0)")
axis.bar(positions + bar_width / 2, class_counts_table["fail"], bar_width, color=COLOR_FAIL, label="Fail (1)")
for position in positions:
    axis.text(position - bar_width / 2, class_counts_table["pass"][position],
              str(class_counts_table["pass"][position]), ha="center", va="bottom", fontsize=9)
    axis.text(position + bar_width / 2, class_counts_table["fail"][position],
              str(class_counts_table["fail"][position]), ha="center", va="bottom", fontsize=9)
axis.set_xticks(positions)
axis.set_xticklabels(["Original", "SMOTE", "ROSE", "ADASYN"])
axis.set_ylabel("Training wafers")
axis.set_title("Class counts before and after balancing (training set only; test set never balanced)")
axis.legend()
axis.spines["top"].set_visible(False)
axis.spines["right"].set_visible(False)
save_figure(figure, "block11_01_class_counts.png")


# ----- 11.2 Screening: five options, same folds, balancing inside each training fold -----
print("\n--- 11.2 Screening five balancing options (random forest, 18 sensors, 5-fold CV) ---")
balancing_options = ["none", "class_weight", "SMOTE", "ROSE", "ADASYN"]
fold_costs = {}
fold_rows_all = []
total_rows = []
for option in balancing_options:
    folds, predictions, probabilities = cross_validate_random_forest(X_train_selected, y_train, option)
    fold_costs[option] = folds["cost_eur"].to_numpy()
    for row_number in range(len(folds)):
        fold_rows_all.append({"option": option, "fold": int(folds["fold"][row_number]),
                              "cost_eur": int(folds["cost_eur"][row_number]),
                              "recall": folds["recall"][row_number],
                              "TP": int(folds["TP"][row_number]), "FP": int(folds["FP"][row_number])})
    total_row = evaluate_predictions(y_train, predictions)
    total_row["option"] = option
    total_row["mean_cost_per_fold"] = round(float(folds["cost_eur"].mean()), 1)
    total_row["std_cost_per_fold"] = round(float(folds["cost_eur"].std()), 1)
    total_rows.append(total_row)

balancing_folds_table = pd.DataFrame(fold_rows_all)
balancing_folds_table.to_csv(TABLES_FOLDER / "block11_balancing_folds.csv", index=False)
balancing_comparison = pd.DataFrame(total_rows)
balancing_comparison = balancing_comparison[["option", "TN", "FP", "FN", "TP", "recall", "precision",
                                             "specificity", "F1", "cost_eur",
                                             "mean_cost_per_fold", "std_cost_per_fold"]]
print(balancing_comparison.to_string(index=False))
balancing_comparison.to_csv(TABLES_FOLDER / "block11_balancing_comparison.csv", index=False)
print("Cost per fold (EUR):")
for option in balancing_options:
    # tolist() turns the numpy array into plain Python numbers for printing.
    print("  " + option.ljust(13), fold_costs[option].tolist())

# Line chart: cost of every option in every fold, so stability is visible.
option_colors = {"none": "#a6a6a6", "class_weight": "#2a78d6", "SMOTE": "#eb6834",
                 "ROSE": "#1baf7a", "ADASYN": "#6250d6"}
fold_numbers = np.arange(1, CV_FOLDS + 1)
figure, axis = plt.subplots(figsize=(8, 4.5))
for option in balancing_options:
    axis.plot(fold_numbers, fold_costs[option] / 1000, marker="o", linewidth=2,
              color=option_colors[option], label=option)
axis.set_xticks(fold_numbers)
axis.set_xlabel("Cross-validation fold")
axis.set_ylabel("Cost in the fold (thousand EUR)")
axis.set_title("Balancing options: cost per fold (random forest, 18 sensors, training set)")
axis.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)
axis.spines["top"].set_visible(False)
axis.spines["right"].set_visible(False)
save_figure(figure, "block11_02_cost_per_fold.png")


# ----- 11.3 Drop rule for ADASYN (fixed in advance) -----
print("\n--- 11.3 ADASYN drop rule ---")
# The best option is the one with the lowest total cross-validated cost.
total_costs = balancing_comparison.set_index("option")["cost_eur"]
best_option = total_costs.idxmin()
print("Best option (lowest total cost):", best_option)

# A neighbour search over the real training wafers (18 sensors), used without
# any model to see where failed wafers lie. Default distance: Euclidean.
neighbour_finder = NearestNeighbors(n_neighbors=5)
neighbour_finder.fit(X_train_selected)

# For context: for each real failed wafer, the share of passed wafers among its
# 5 nearest other wafers (6 neighbours are asked for, because the nearest one
# is the wafer itself at distance 0 and is skipped).
distances, neighbour_positions = neighbour_finder.kneighbors(X_train_selected[y_train == 1], n_neighbors=6)
real_pass_neighbours = (y_train.to_numpy()[neighbour_positions[:, 1:]] == 0).mean()
print("Real failed wafers: average share of passed wafers among their 5 nearest neighbours:",
      round(float(real_pass_neighbours), 3))

if best_option == "ADASYN":
    condition_1 = False
    condition_2 = False
    condition_3 = False
    print("ADASYN is itself the best option, so it cannot be dropped.")
else:
    # Fold-by-fold cost differences ADASYN minus best option (positive = ADASYN worse).
    differences = fold_costs["ADASYN"] - fold_costs[best_option]
    worse_folds = int((differences > 0).sum())
    mean_difference = float(differences.mean())
    std_difference = float(differences.std(ddof=1))
    # Condition 1: worse in at least 4 of the 5 folds.
    condition_1 = worse_folds >= 4
    # Condition 2: worse on average by more than the fold-to-fold noise
    # (sample standard deviation of the differences, ddof=1).
    condition_2 = mean_difference > std_difference
    print("Condition 1 - worse in at least 4 of 5 folds:", condition_1, "(worse in", worse_folds, "folds)")
    print("Condition 2 - mean difference > std of differences:", condition_2,
          "(mean", round(mean_difference), "EUR, std", round(std_difference), "EUR)")

    # Condition 3, measured without any model: where do the synthetic failed
    # wafers land? For each synthetic wafer, count how many of its 5 nearest
    # real training wafers are passed wafers. If ADASYN puts more synthetic
    # failures deep inside passed regions (4 or 5 of 5 neighbours passed) than
    # SMOTE does, the reason would come from the data and apply to every model.
    deep_in_pass_share = {}
    for option in ["SMOTE", "ADASYN"]:
        X_balanced, y_balanced, unused_weight = balance_training_data(option, X_train_selected, y_train)
        synthetic_rows = X_balanced.iloc[len(X_train_selected):]
        distances, neighbour_positions = neighbour_finder.kneighbors(synthetic_rows)
        pass_neighbour_counts = (y_train.to_numpy()[neighbour_positions] == 0).sum(axis=1)
        deep_in_pass_share[option] = float((pass_neighbour_counts >= 4).mean())
        print("  " + option + ": share of synthetic failed wafers with >= 4 of 5 passed neighbours:",
              round(deep_in_pass_share[option], 3))
    condition_3 = deep_in_pass_share["ADASYN"] > deep_in_pass_share["SMOTE"]
    print("Condition 3 - data-based reason (ADASYN deeper in passed regions than SMOTE):", condition_3)

drop_adasyn = condition_1 and condition_2 and condition_3
options_for_phase3 = list(balancing_options)
if drop_adasyn:
    options_for_phase3.remove("ADASYN")
print("ADASYN dropped:", drop_adasyn, "| options continuing to Phase 3:", options_for_phase3)


# ----- 11.4 Working choice for Phase 2 (decision D11) -----
print("\n--- 11.4 Working choice (D11) ---")
print("Setting BALANCING_METHOD:", BALANCING_METHOD, "| screening winner:", best_option)
if BALANCING_METHOD != best_option:
    print("Note: the setting differs from the screening winner; check decision D11.")

# Processed data: the whole training set balanced with the working choice
# (original wafers keep their wafer_id; new wafers are named synthetic_...).
X_train_balanced, y_train_balanced, working_class_weight = balance_training_data(
    BALANCING_METHOD, X_train_selected, y_train)
print("Training set after", BALANCING_METHOD + ":", int((y_train_balanced == 0).sum()), "pass /",
      int((y_train_balanced == 1).sum()), "fail")
save_processed(X_train_balanced, y_train_balanced, "block11_train_balanced_" + BALANCING_METHOD.lower() + ".csv")

# %% Block 12: Evaluation helpers and detailed assessment of the current best model
# Functions that turn true labels, predicted labels and predicted failure
# probabilities into the confusion matrix (as a figure), accuracy, recall,
# specificity, precision, F1, F2, ROC-AUC with the ROC curve, and total cost
# (manual week 4 step 4; Section 13). They are applied to the cross-validated
# predictions of the current best model: random forest + 18 Boruta sensors +
# ROSE inside each training fold. The test set stays locked (used once, Phase 3).


def full_evaluation(y_true, y_pred, probabilities):
    """
    All metrics for one set of predictions: the confusion-matrix values,
    accuracy, recall, specificity, precision, F1 and cost from
    evaluate_predictions (Block 9), plus F2 and ROC-AUC.

    Input:  y_true (true labels, 1 = fail), y_pred (predicted labels),
            probabilities (predicted failure probabilities, same order)
    Output: dictionary with all values
    """
    result = evaluate_predictions(y_true, y_pred)
    precision = result["precision"]
    recall = result["recall"]
    # F2 weights recall twice as much as precision: 5PR / (4P + R).
    if 4 * precision + recall > 0:
        result["F2"] = round(5 * precision * recall / (4 * precision + recall), 4)
    else:
        result["F2"] = 0.0
    # ROC-AUC uses the probabilities, not the 0.5 decision: the chance that a
    # random failed wafer gets a higher failure probability than a random passed one.
    result["ROC_AUC"] = round(float(roc_auc_score(y_true, probabilities)), 4)
    return result


def plot_confusion_matrix(y_true, y_pred, title, file_name):
    """
    Draw a labelled confusion matrix (rows = true class, columns = predicted
    class) with the count and the cost of each cell, and save it.

    Input:  y_true (true labels), y_pred (predicted labels), title (str), file_name (str)
    Output: none (the figure is saved)
    """
    result = evaluate_predictions(y_true, y_pred)
    matrix = np.array([[result["TN"], result["FP"]], [result["FN"], result["TP"]]])
    cell_names = [["TN\ngood wafer passed", "FP (Type I error)\ngood wafer flagged"],
                  ["FN (Type II error)\nfailed wafer missed", "TP\nfailed wafer caught"]]
    cell_costs = [[0, COST_FALSE_POSITIVE], [COST_FALSE_NEGATIVE, 0]]
    figure, axis = plt.subplots(figsize=(7, 5.5))
    # One blue scale from 0 to the number of wafers, so cells are comparable.
    axis.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.sum())
    for row in range(2):
        for column in range(2):
            count = matrix[row, column]
            cell_cost = count * cell_costs[row][column]
            text_color = "white" if count > matrix.sum() / 2 else "#222222"
            axis.text(column, row, cell_names[row][column] + "\n" + str(count)
                      + "\ncost EUR " + format(int(cell_cost), ","),
                      ha="center", va="center", fontsize=10, color=text_color)
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["Predicted pass (0)", "Predicted fail (1)"])
    axis.set_yticks([0, 1])
    axis.set_yticklabels(["True pass (0)", "True fail (1)"])
    axis.set_title(title + "\nrecall " + str(result["recall"]) + " | precision " + str(result["precision"])
                   + " | total cost EUR " + format(result["cost_eur"], ","), fontsize=10)
    save_figure(figure, file_name)


def plot_roc_curve(y_true, probabilities, y_pred, title, file_name):
    """
    Draw the ROC curve (true-positive rate against false-positive rate over
    all thresholds) with its AUC, the diagonal of a random guess, and the
    point of the actual predictions, and save it.

    Input:  y_true (true labels), probabilities (predicted failure probabilities),
            y_pred (predicted labels, marked as one point), title (str), file_name (str)
    Output: none (the figure is saved)
    """
    # roc_curve() tries every threshold and returns the false-positive rate
    # (1 - specificity) and the true-positive rate (recall) for each.
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, probabilities)
    auc = roc_auc_score(y_true, probabilities)
    result = evaluate_predictions(y_true, y_pred)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot(false_positive_rate, true_positive_rate, linewidth=2, color=COLOR_MAIN,
              label="Model (AUC = " + str(round(auc, 3)) + ")")
    axis.plot([0, 1], [0, 1], linestyle="--", color=COLOR_REFERENCE, label="Random guess (AUC = 0.5)")
    axis.plot(1 - result["specificity"], result["recall"], marker="o", markersize=9, color=COLOR_FAIL,
              linestyle="none", label="Threshold 0.5 (recall " + str(result["recall"]) + ")")
    axis.set_xlabel("False-positive rate (good wafers flagged)")
    axis.set_ylabel("True-positive rate = recall (failed wafers caught)")
    axis.set_title(title, fontsize=10)
    axis.legend(loc="lower right")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    save_figure(figure, file_name)


# ----- 12.1 Cross-validated predictions of the current best model -----
print("\n--- 12.1 Detailed assessment: random forest + 18 sensors + " + BALANCING_METHOD + " (5-fold CV) ---")
# Same function, same five parts as Blocks 9 to 11. Every probability comes
# from a model that did not see that wafer; balancing only in the training parts.
best_folds, best_predictions, best_probabilities = cross_validate_random_forest(
    X_train_selected, y_train, BALANCING_METHOD)
no_balancing_folds, no_balancing_predictions, no_balancing_probabilities = cross_validate_random_forest(
    X_train_selected, y_train, "none")

metric_rows = []
for name, predictions, probabilities in [
        ("RF + 18 sensors, no balancing", no_balancing_predictions, no_balancing_probabilities),
        ("RF + 18 sensors + " + BALANCING_METHOD, best_predictions, best_probabilities)]:
    row = full_evaluation(y_train, predictions, probabilities)
    row["model"] = name
    metric_rows.append(row)
metrics_table = pd.DataFrame(metric_rows)
metrics_table = metrics_table[["model", "TN", "FP", "FN", "TP", "accuracy", "recall", "specificity",
                               "precision", "F1", "F2", "ROC_AUC", "cost_eur"]]
print(metrics_table.to_string(index=False))
# Build the file name first: "/" (join a folder and a name) is applied before
# "+", so joining and adding text in one expression would fail.
metrics_file_name = "block12_metrics_" + BALANCING_METHOD.lower() + ".csv"
metrics_table.to_csv(TABLES_FOLDER / metrics_file_name, index=False)
print("For comparison - always 'pass': accuracy", round(float((y_train == 0).mean()), 4),
      "| recall 0 | cost EUR", format(int(COST_FALSE_NEGATIVE * (y_train == 1).sum()), ","))

plot_confusion_matrix(y_train, best_predictions,
                      "Random forest + 18 Boruta sensors + " + BALANCING_METHOD + " (5-fold CV, training set)",
                      "block12_01_confusion_matrix_" + BALANCING_METHOD.lower() + ".png")
plot_roc_curve(y_train, best_probabilities, best_predictions,
               "ROC curve: random forest + 18 sensors + " + BALANCING_METHOD + " (5-fold CV, training set)",
               "block12_02_roc_curve_" + BALANCING_METHOD.lower() + ".png")


# ----- 12.2 The precision-recall trade-off (illustration only, no threshold chosen) -----
print("\n--- 12.2 Precision-recall trade-off at other thresholds (illustration; D16 is decided in Phase 3) ---")
# A wafer is called "fail" when its failure probability is above the threshold.
# At 0.5 this is exactly what predict() does (more than half of the trees vote fail).
threshold_rows = []
for threshold in [0.5, 0.4, 0.3, 0.2]:
    threshold_predictions = (best_probabilities > threshold).astype(int)
    row = evaluate_predictions(y_train, threshold_predictions)
    row["threshold"] = threshold
    threshold_rows.append(row)
threshold_table = pd.DataFrame(threshold_rows)
threshold_table = threshold_table[["threshold", "TP", "FN", "FP", "recall", "precision", "cost_eur"]]
print(threshold_table.to_string(index=False))
threshold_table.to_csv(TABLES_FOLDER / "block12_threshold_illustration.csv", index=False)


# ----- 12.3 Normality check (manual week 5, step 1) -----
print("\n--- 12.3 Normality check ---")
# A Box-Cox or Yeo-Johnson transformation is only needed if the model assumes
# normally distributed inputs. A random forest does not (it only compares values
# with thresholds), so no transformation is applied. For the record, the
# skewness of the 18 selected sensors (0 = symmetric like a bell curve; skewness
# does not change with min-max scaling):
selected_skewness = X_train_selected.skew().round(3)
selected_skewness.to_csv(TABLES_FOLDER / "block12_selected_sensor_skewness.csv", header=["skewness"],
                         index_label="sensor")
print("Selected sensors with |skewness| > 1 (clearly not bell-shaped):",
      int((selected_skewness.abs() > 1).sum()), "of", len(selected_skewness))
print("Skewness range:", selected_skewness.min(), "to", selected_skewness.max())
print("Conclusion: random forest does not assume normality -> no Box-Cox / Yeo-Johnson.")


# %% Block 13: Process map and the leak-proof pipeline (training set only)
# Phase 2 fitted the scaler and the imputer once on the whole training set and
# then cross-validated. Here every step that learns from data (scaling, kNN
# imputation, balancing, model) is put into one imbalanced-learn Pipeline, so
# each cross-validation fold refits all of them on its own training part only
# (manual Section 10). The rules decided once in Phase 1 (140 removed columns,
# 3 sigma bounds) are applied before the pipeline (manual 10.2), and the 18
# Boruta sensors stay a fixed list (decision of 20 Oct; one in-fold Boruta check
# follows in Block 15). The test set is not used in this block.


def select_sensor_columns(data, positions):
    """
    Keep only the columns at the given positions (the 18 Boruta sensors).
    Used inside the pipeline, after scaling and imputation, which pass on
    plain arrays without column names.

    Input:  data (2-D array), positions (list of column positions to keep)
    Output: 2-D array with only those columns
    """
    return data[:, positions]


def make_sampler(option):
    """
    The balancing step for the pipeline. "none" and "class_weight" add no
    wafers, so the step is skipped ("passthrough"); class weights are set on
    the model instead.

    Input:  option (str: "none", "class_weight", "SMOTE", "ROSE" or "ADASYN")
    Output: an imbalanced-learn sampler, or "passthrough"
    """
    if option == "none" or option == "class_weight":
        return "passthrough"
    if option == "SMOTE":
        return SMOTE(random_state=SEED)
    if option == "ROSE":
        return RandomOverSampler(shrinkage=ROSE_SHRINKAGE, random_state=SEED)
    if option == "ADASYN":
        return ADASYN(random_state=SEED)
    raise ValueError("Unknown balancing option: " + str(option))


def build_pipeline(model, balancing, n_neighbors=KNN_NEIGHBORS):
    """
    Build the leak-proof pipeline: min-max scaling -> kNN imputation ->
    the 18 Boruta sensors -> balancing -> model. Every step is refitted
    whenever the pipeline is fitted, so in cross-validation each fold learns
    only from its own training part.

    Input:  model (an unfitted classifier), balancing (str, see make_sampler),
            n_neighbors (int, k for the kNN imputer; default KNN_NEIGHBORS)
    Output: an imbalanced-learn Pipeline (not yet fitted)
    """
    # imbalanced-learn's Pipeline (not scikit-learn's) is needed because the
    # balancer ADDS rows; it also skips the balancer at prediction time, so the
    # wafers being evaluated are never balanced.
    return Pipeline([
        ("scaler", MinMaxScaler()),
        ("imputer", KNNImputer(n_neighbors=n_neighbors)),
        ("selector", FunctionTransformer(select_sensor_columns, kw_args={"positions": selected_positions})),
        ("balancer", make_sampler(balancing)),
        ("model", model),
    ])


def cross_validate_pipeline(pipeline, X_data, y_data):
    """
    Cross-validate a pipeline with the same parts as in Phase 2 (cv_splitter):
    for each part, a fresh copy of the pipeline (clone) is fitted on the other
    parts and predicts this one.

    Input:  pipeline (unfitted Pipeline), X_data (DataFrame), y_data (Series)
    Output: fold_costs (array: cost per part), cv_probabilities (Series:
            out-of-fold failure probability per wafer)
    """
    cv_probabilities = pd.Series(0.0, index=y_data.index)
    fold_costs = []
    for train_positions, validation_positions in cv_splitter.split(X_data, y_data):
        # clone() copies the settings without anything learned, so no fold
        # reuses what another fold learned.
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_data.iloc[train_positions], y_data.iloc[train_positions])
        fold_probabilities = fold_pipeline.predict_proba(X_data.iloc[validation_positions])[:, 1]
        cv_probabilities.iloc[validation_positions] = fold_probabilities
        # Cost of this part at the default rule (fail if probability above 0.5).
        fold_predictions = (fold_probabilities > 0.5).astype(int)
        fold_result = evaluate_predictions(y_data.iloc[validation_positions], fold_predictions)
        fold_costs.append(fold_result["cost_eur"])
    return np.array(fold_costs), cv_probabilities


# The input to every pipeline: the training set after the rules applied once
# (450 columns, capped, gaps still present). The 18 sensors are found by name.
X_pipeline_input = X_train_treated
selected_positions = []
for sensor in selected_features:
    selected_positions.append(list(X_pipeline_input.columns).index(sensor))


# ----- 13.1 Process map (manual week 5, step 6) -----
# Boxes in three colours: grey = exploratory only; orange = learned once on the
# training set, applied unchanged to the test set; green = refitted inside every
# cross-validation fold (the pipeline); blue = the locked test-set path.
map_colors = {"explore": "#d9d9d9", "once": "#fbd6c2", "pipeline": "#cfe8d3", "test": "#cfe0f5"}
map_boxes = [
    (5.0, 9.4, "Download from UCI (Block 2): names, labels 0/1, time stamps", "once"),
    (1.6, 8.3, "Block 3: descriptive analysis\n(full data, exploratory only)", "explore"),
    (5.0, 8.3, "Block 4: stratified 80/20 split (seed 42)", "once"),
    (3.0, 7.1, "Training set: 1,253 wafers (83 failed)", "once"),
    (7.6, 7.1, "Test set: 314 wafers (21 failed) - LOCKED", "test"),
    (3.0, 6.0, "Block 5: remove 140 columns\n(rule learned on training)", "once"),
    (7.6, 6.0, "Remove the same 140 columns", "test"),
    (3.0, 4.9, "Block 7: cap at training 3-sigma bounds", "once"),
    (7.6, 4.9, "Cap with the training bounds", "test"),
    (3.0, 3.55, "PIPELINE, refitted in every CV fold:\nmin-max scaling -> kNN imputation (k = 5)\n"
                "-> 18 Boruta sensors -> balancing (training part only) -> model", "pipeline"),
    (7.6, 3.55, "Fitted pipeline: transform only\n(no fitting, no balancing)", "test"),
    (3.0, 2.2, "Blocks 14-16: model comparison, grid search,\nthreshold by cost (cross-validation)", "pipeline"),
    (7.6, 2.2, "Block 17: predict with the chosen\nthreshold - FINAL TEST, ONCE", "test"),
]
figure, axis = plt.subplots(figsize=(12, 8))
axis.set_xlim(0, 10)
axis.set_ylim(1.4, 10)
axis.axis("off")
for x_position, y_position, text, group in map_boxes:
    axis.text(x_position, y_position, text, ha="center", va="center", fontsize=9,
              bbox={"boxstyle": "round,pad=0.5", "facecolor": map_colors[group], "edgecolor": "#7f7f7f"})
# Arrows (start -> end) along the training path, the test path and the split.
map_arrows = [((5.0, 9.15), (5.0, 8.55)), ((4.4, 9.15), (1.6, 8.65)),
              ((5.0, 8.05), (3.0, 7.35)), ((5.0, 8.05), (7.6, 7.35)),
              ((3.0, 6.85), (3.0, 6.3)), ((3.0, 5.7), (3.0, 5.15)), ((3.0, 4.65), (3.0, 4.05)),
              ((3.0, 3.05), (3.0, 2.55)), ((7.6, 6.85), (7.6, 6.25)), ((7.6, 5.75), (7.6, 5.15)),
              ((7.6, 4.65), (7.6, 3.9)), ((7.6, 3.2), (7.6, 2.55)), ((5.35, 3.55), (6.3, 3.55))]
for start, end in map_arrows:
    axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#595959"})
legend_handles = [Patch(color=map_colors["explore"], label="exploratory only (full data)"),
                  Patch(color=map_colors["once"], label="learned once on the training set"),
                  Patch(color=map_colors["pipeline"], label="refitted inside every CV fold"),
                  Patch(color=map_colors["test"], label="test set: only applies stored rules")]
axis.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
axis.set_title("Process map: what is learned where (training set) and only applied (test set)")
save_figure(figure, "block13_01_process_map.png")


# ----- 13.2 The pipeline with the Phase 2 working choice, cross-validated -----
print("\n--- 13.2 Leak-proof pipeline: random forest + 18 sensors + " + BALANCING_METHOD + " (5-fold CV) ---")
working_pipeline = build_pipeline(RandomForestClassifier(random_state=SEED), BALANCING_METHOD)
pipeline_fold_costs, pipeline_probabilities = cross_validate_pipeline(working_pipeline, X_pipeline_input, y_train)
pipeline_predictions = (pipeline_probabilities > 0.5).astype(int)

# Phase 2 chain (scaler and imputer fitted once on the whole training set, from
# Block 12) vs. the pipeline (refitted in every fold): the difference shows how
# optimistic the simpler Phase 2 set-up was.
leak_rows = []
for name, predictions, probabilities in [
        ("Phase 2 chain (scaler + imputer fitted once)", best_predictions, best_probabilities),
        ("Leak-proof pipeline (refitted per fold)", pipeline_predictions, pipeline_probabilities)]:
    row = full_evaluation(y_train, predictions, probabilities)
    row["set_up"] = name
    leak_rows.append(row)
pipeline_comparison = pd.DataFrame(leak_rows)
pipeline_comparison = pipeline_comparison[["set_up", "TN", "FP", "FN", "TP", "recall", "precision",
                                           "F2", "ROC_AUC", "cost_eur"]]
print(pipeline_comparison.to_string(index=False))
pipeline_comparison.to_csv(TABLES_FOLDER / "block13_pipeline_vs_phase2.csv", index=False)
print("Cost per fold (pipeline):", pipeline_fold_costs.tolist())


# ----- 13.3 Pipeline checks (manual 10.5, adapted: the test set stays locked) -----
print("\n--- 13.3 Pipeline checks ---")
# Shape check: all steps except the model. fit_resample runs scaling,
# imputation, sensor selection and balancing on the training set.
check_pipeline = clone(working_pipeline)
X_check, y_check = check_pipeline[:-1].fit_resample(X_pipeline_input, y_train)
print("Shape just before the model:", X_check.shape, "| pass", int((y_check == 0).sum()),
      "| fail", int((y_check == 1).sum()), "(expected 2340 x 18 with ROSE)")

# Reproducibility check: a second cross-validation run must give exactly the
# same probabilities (every random step has random_state=SEED).
repeat_fold_costs, repeat_probabilities = cross_validate_pipeline(working_pipeline, X_pipeline_input, y_train)
print("Second run identical:", bool(np.array_equal(pipeline_probabilities.to_numpy(), repeat_probabilities.to_numpy())))

# Row counts side by side: the pipeline is only ever fitted on training wafers.
print("Training wafers used for fitting:", len(X_pipeline_input), "| test wafers (untouched):", len(X_test))


# %% Block 14: Model x balancing comparison (training set only)
# Compares model families and balancing options fairly: the same leak-proof
# pipeline, the same 18 sensors and the same five CV parts for every
# combination (decisions D11, D13). The shortlist for the grid search follows
# the rule fixed before this run (DECISIONS.md, 20 Oct): rank by mean average
# precision (D15) and take the best combination of each model family until 5
# families are chosen. Optional soft-voting check (D17). The test set is not used.


def make_model(family, balancing):
    """
    Create an unfitted model of one family. With balancing "class_weight" the
    model weights each failed wafer by about 1,170 / 83 (class_weight="balanced").

    Input:  family (str), balancing (str)
    Output: an unfitted classifier
    """
    class_weight = "balanced" if balancing == "class_weight" else None
    if family == "logistic_regression":
        # max_iter=1000 (default 100) gives the solver enough steps to finish;
        # default C=1.0 (strength of the penalty), tuned in Block 15.
        return LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=SEED)
    if family == "random_forest":
        # Defaults: 100 trees, max_features="sqrt".
        return RandomForestClassifier(class_weight=class_weight, random_state=SEED)
    if family == "gradient_boosting":
        # Defaults: 100 boosting rounds, learning_rate=0.1, max_leaf_nodes=31;
        # it can handle missing values itself, but gets the imputed data here.
        return HistGradientBoostingClassifier(class_weight=class_weight, random_state=SEED)
    if family == "naive_bayes":
        # Gaussian naive Bayes has no class weights and no randomness.
        return GaussianNB()
    if family == "balanced_random_forest":
        # Each of its 100 trees learns from all failed wafers plus an equally
        # large random sample of passed wafers (defaults sampling_strategy="all",
        # replacement=True, bootstrap=False): balancing happens inside the model.
        return BalancedRandomForestClassifier(random_state=SEED)
    if family == "easy_ensemble":
        # 10 AdaBoost models (default), each on all failed wafers plus a random,
        # equally large sample of passed wafers: balancing inside the model.
        return EasyEnsembleClassifier(random_state=SEED)
    if family == "dummy":
        # Always predicts the most frequent class ("pass"): the no-skill baseline.
        return DummyClassifier(strategy="most_frequent")
    raise ValueError("Unknown model family: " + str(family))


def fold_values(y_data, probabilities):
    """
    Average precision and recall (at threshold 0.5) in each CV part, from the
    out-of-fold probabilities (cv_splitter always gives the same parts).

    Input:  y_data (Series of labels), probabilities (Series, out-of-fold)
    Output: two arrays: average precision per part, recall per part
    """
    precision_values = []
    recall_values = []
    for train_positions, validation_positions in cv_splitter.split(y_data, y_data):
        y_part = y_data.iloc[validation_positions]
        probabilities_part = probabilities.iloc[validation_positions]
        precision_values.append(average_precision_score(y_part, probabilities_part))
        recall_values.append(evaluate_predictions(y_part, (probabilities_part > 0.5).astype(int))["recall"])
    return np.array(precision_values), np.array(recall_values)


# ----- 14.1 The combinations -----
# Every model family with the balancing options it can use; the two balanced
# ensembles and the dummy balance internally or not at all ("none" only).
combinations = []
for family in ["logistic_regression", "random_forest", "gradient_boosting"]:
    for balancing in options_for_phase3:
        combinations.append((family, balancing))
for balancing in options_for_phase3:
    if balancing != "class_weight":
        combinations.append(("naive_bayes", balancing))
combinations.append(("balanced_random_forest", "none"))
combinations.append(("easy_ensemble", "none"))
combinations.append(("dummy", "none"))
print("\n--- 14.1 Model x balancing matrix:", len(combinations), "combinations, 5-fold CV ---")

matrix_rows = []
matrix_fold_rows = []
combination_probabilities = {}
combination_average_precisions = {}
for family, balancing in combinations:
    pipeline = build_pipeline(make_model(family, balancing), balancing)
    costs_per_fold, probabilities = cross_validate_pipeline(pipeline, X_pipeline_input, y_train)
    precisions_per_fold, recalls_per_fold = fold_values(y_train, probabilities)
    name = family + " + " + balancing
    combination_probabilities[name] = probabilities
    combination_average_precisions[name] = precisions_per_fold
    totals = full_evaluation(y_train, (probabilities > 0.5).astype(int), probabilities)
    matrix_rows.append({"combination": name, "family": family, "balancing": balancing,
                        "mean_average_precision": round(float(precisions_per_fold.mean()), 4),
                        "std_average_precision": round(float(precisions_per_fold.std(ddof=1)), 4),
                        "ROC_AUC": totals["ROC_AUC"],
                        "mean_recall_at_0.5": round(float(recalls_per_fold.mean()), 4),
                        "TP_at_0.5": totals["TP"], "FP_at_0.5": totals["FP"],
                        "cost_at_0.5": totals["cost_eur"],
                        "std_cost_per_fold": round(float(costs_per_fold.std(ddof=1)), 1)})
    for part in range(len(costs_per_fold)):
        matrix_fold_rows.append({"combination": name, "fold": part + 1,
                                 "average_precision": round(float(precisions_per_fold[part]), 4),
                                 "recall_at_0.5": recalls_per_fold[part],
                                 "cost_at_0.5": int(costs_per_fold[part])})
    print("  " + name.ljust(40), "AP", round(float(precisions_per_fold.mean()), 3),
          "+/-", round(float(precisions_per_fold.std(ddof=1)), 3), "| cost at 0.5", totals["cost_eur"])

model_matrix = pd.DataFrame(matrix_rows).sort_values("mean_average_precision", ascending=False)
model_matrix.to_csv(TABLES_FOLDER / "block14_model_matrix.csv", index=False)
pd.DataFrame(matrix_fold_rows).to_csv(TABLES_FOLDER / "block14_model_matrix_folds.csv", index=False)


# ----- 14.2 Shortlist for the grid search (rule fixed before this run) -----
print("\n--- 14.2 Shortlist: best combination of each family by mean average precision (5 families) ---")
shortlist_rows = []
families_taken = []
for row_number in range(len(model_matrix)):
    row = model_matrix.iloc[row_number]
    if row["family"] == "dummy" or row["family"] in families_taken:
        continue
    families_taken.append(row["family"])
    shortlist_rows.append(row)
    if len(families_taken) == 5:
        break
shortlist = pd.DataFrame(shortlist_rows)
print(shortlist[["combination", "mean_average_precision", "std_average_precision", "ROC_AUC",
                 "cost_at_0.5"]].to_string(index=False))
shortlist.to_csv(TABLES_FOLDER / "block14_shortlist.csv", index=False)
shortlist_pairs = []
for row_number in range(len(shortlist)):
    shortlist_pairs.append((shortlist.iloc[row_number]["family"], shortlist.iloc[row_number]["balancing"]))

# Chart: mean average precision +/- standard deviation for every combination,
# shortlisted pairs in green, the dummy baseline in grey, the rest in blue.
chart_data = model_matrix.iloc[::-1]
bar_colors = []
for row_number in range(len(chart_data)):
    name = chart_data.iloc[row_number]["combination"]
    if chart_data.iloc[row_number]["family"] == "dummy":
        bar_colors.append("#a6a6a6")
    elif name in list(shortlist["combination"]):
        bar_colors.append("#008300")
    else:
        bar_colors.append(COLOR_MAIN)
figure, axis = plt.subplots(figsize=(9, 9))
axis.barh(chart_data["combination"], chart_data["mean_average_precision"],
          xerr=chart_data["std_average_precision"], color=bar_colors, capsize=3)
axis.axvline(float(y_train.mean()), linestyle="--", color=COLOR_REFERENCE)
axis.text(float(y_train.mean()), -0.9, " random guess = failure share (" + str(round(float(y_train.mean()), 3)) + ")",
          color=COLOR_REFERENCE, fontsize=8, va="top")
legend_handles = [Patch(color="#008300", label="shortlisted (best of 5 families)"),
                  Patch(color=COLOR_MAIN, label="other combinations"),
                  Patch(color="#a6a6a6", label="dummy baseline")]
axis.legend(handles=legend_handles, loc="lower right", fontsize=8)
axis.set_xlabel("Average precision, mean over 5 folds (bars: +/- one standard deviation)")
axis.set_title("Model x balancing comparison (training set, leak-proof pipeline, 18 sensors)")
axis.spines["top"].set_visible(False)
axis.spines["right"].set_visible(False)
save_figure(figure, "block14_01_model_comparison.png")


# ----- 14.3 Optional soft-voting ensemble of the 3 best shortlisted pairs (D17) -----
print("\n--- 14.3 Soft-voting check (D17) ---")
# VotingClassifier with voting="soft" averages the failure probabilities of its
# members. Each member is a complete pipeline with its own balancing, so every
# member is still refitted inside each fold.
voting_members = []
for family, balancing in shortlist_pairs[:3]:
    voting_members.append((family, build_pipeline(make_model(family, balancing), balancing)))
voting_ensemble = VotingClassifier(estimators=voting_members, voting="soft")
voting_costs, voting_probabilities = cross_validate_pipeline(voting_ensemble, X_pipeline_input, y_train)
voting_precisions, voting_recalls = fold_values(y_train, voting_probabilities)

best_single = shortlist.iloc[0]["combination"]
differences = voting_precisions - combination_average_precisions[best_single]
keep_voting = float(differences.mean()) > float(differences.std(ddof=1))
voting_check = pd.DataFrame([{"ensemble": "soft vote of " + " | ".join(shortlist["combination"][:3]),
                              "mean_average_precision": round(float(voting_precisions.mean()), 4),
                              "best_single": best_single,
                              "best_single_average_precision": round(float(combination_average_precisions[best_single].mean()), 4),
                              "mean_difference": round(float(differences.mean()), 4),
                              "std_of_differences": round(float(differences.std(ddof=1)), 4),
                              "kept": keep_voting}])
print(voting_check.T.to_string(header=False))
voting_check.to_csv(TABLES_FOLDER / "block14_voting_check.csv", index=False)
print("Soft-voting ensemble kept (D17):", keep_voting)


# %% Block 15: Grid search on the five shortlisted pairs (training set only)
# Tunes a few settings of each shortlisted model-balancing pair from Block 14:
# the outlier treatment (three versions of the training data, all made with the
# stored Block 7 bounds), k of the kNN imputer and one setting of the model.
# Every setting is scored on the same five CV parts by mean average precision
# (D15); F2 at threshold 0.5 is reported alongside. Grid search rather than
# randomised search, because this grid is small enough to try in full (D14).
# The final model is not chosen here: all five tuned pairs go to the threshold
# step (Block 16). A last check refits Boruta inside every fold for the best
# pair (manual 7.5). Manual Section 11. The test set is not used in this block.


def fold_scores_from_search(search_results, row_number, score_name):
    """
    Read the five fold scores of one setting from a fitted GridSearchCV.

    Input:  search_results (dict: cv_results_ of GridSearchCV), row_number (int:
            which setting), score_name (str: "average_precision" or "F2")
    Output: array with one score per CV part
    """
    scores = []
    for part in range(CV_FOLDS):
        scores.append(search_results["split" + str(part) + "_test_" + score_name][row_number])
    return np.array(scores)


def cross_validate_with_boruta(X_data, y_data, family, balancing, n_neighbors, setting_name, setting_value):
    """
    Cross-validate one tuned pair with Boruta refitted inside every training
    part, instead of the fixed 18 sensors, to measure how optimistic the fixed
    list made the scores (manual 7.5).

    Input:  X_data (DataFrame, one outlier version), y_data (Series), family and
            balancing (str), n_neighbors (int), setting_name (str) and
            setting_value (the tuned model setting)
    Output: precisions (array: average precision per part), sensor_counts and
            shared_counts (lists: sensors chosen per part, and how many of them
            are among the fixed 18)
    """
    precisions = []
    sensor_counts = []
    shared_counts = []
    for train_positions, validation_positions in cv_splitter.split(X_data, y_data):
        X_part_train = X_data.iloc[train_positions]
        y_part_train = y_data.iloc[train_positions]
        X_part_validation = X_data.iloc[validation_positions]
        y_part_validation = y_data.iloc[validation_positions]
        # Scaling and imputation are learned on this training part only and then
        # applied to both parts, exactly as the pipeline does.
        scaler_part = MinMaxScaler().fit(X_part_train)
        imputer_part = KNNImputer(n_neighbors=n_neighbors).fit(scaler_part.transform(X_part_train))
        train_ready = imputer_part.transform(scaler_part.transform(X_part_train))
        validation_ready = imputer_part.transform(scaler_part.transform(X_part_validation))
        # Boruta with the Block 10 settings, fitted on this training part only.
        forest_part = RandomForestClassifier(class_weight="balanced", max_depth=5,
                                             n_jobs=None, random_state=SEED)
        boruta_part = BorutaPy(forest_part, n_estimators="auto", max_iter=BORUTA_MAX_ITER,
                               random_state=SEED, verbose=0)
        boruta_part.fit(train_ready, y_part_train.to_numpy())
        # Confirmed sensors, plus tentative ones if KEEP_TENTATIVE (D10);
        # "|" combines the two True/False lists ("or" per sensor).
        keep = boruta_part.support_.copy()
        if KEEP_TENTATIVE:
            keep = keep | boruta_part.support_weak_
        positions = np.where(keep)[0]
        if len(positions) == 0:
            raise ValueError("Boruta kept no sensor in one fold.")
        # Balancing (training part only) and the tuned model. set_params takes
        # the setting by name; ** turns the small dictionary into name=value.
        model = make_model(family, balancing)
        model.set_params(**{setting_name: setting_value})
        rest_of_pipeline = Pipeline([("balancer", make_sampler(balancing)), ("model", model)])
        rest_of_pipeline.fit(train_ready[:, positions], y_part_train)
        probabilities = rest_of_pipeline.predict_proba(validation_ready[:, positions])[:, 1]
        precisions.append(average_precision_score(y_part_validation, probabilities))
        # How many of the chosen sensors are also among the fixed 18?
        shared = 0
        for position in positions:
            if X_data.columns[position] in selected_features:
                shared = shared + 1
        sensor_counts.append(len(positions))
        shared_counts.append(shared)
    return np.array(precisions), sensor_counts, shared_counts


# ----- 15.1 The search space (fixed before the run, DECISIONS.md 20 Oct) -----
# The three outlier options of Block 7 (D7); "cap" is the current choice.
GRID_OUTLIER_OPTIONS = ["nothing", "cap", "remove"]
# k of the kNN imputer: the manual's 5 and one step below and above.
GRID_NEIGHBORS = [3, 5, 7]
# One setting per model family: (name, three values, default used in Block 14).
# Each list contains the default, so tuning can always fall back to it.
MODEL_SETTINGS = {
    # Number of balanced AdaBoost sub-models: more sub-models see more of the
    # 1,170 passed wafers.
    "easy_ensemble": ("n_estimators", [10, 25, 50], 10),
    # Fewest wafers allowed in a leaf: larger leaves stop the trees memorising
    # single (or copied) failed wafers.
    "random_forest": ("min_samples_leaf", [1, 5, 10], 1),
    "balanced_random_forest": ("min_samples_leaf", [1, 5, 10], 1),
    # Size of each boosting correction: smaller steps learn more carefully.
    "gradient_boosting": ("learning_rate", [0.03, 0.1, 0.3], 0.1),
    # Inverse penalty strength: a smaller C gives a simpler, steadier boundary.
    "logistic_regression": ("C", [0.01, 0.1, 1.0], 1.0),
}


# ----- 15.2 Three versions of the training data (outlier options) -----
print("\n--- 15.2 Outlier versions of the training set (stored Block 7 bounds) ---")
outlier_versions = {}
for option in GRID_OUTLIER_OPTIONS:
    outlier_versions[option] = apply_outlier_treatment(option, X_train_reduced, lower_bounds, upper_bounds)
    print("  " + option.ljust(8), "shape", outlier_versions[option].shape,
          "| gaps for the imputer:", int(outlier_versions[option].isna().sum().sum()))
# The "cap" version must be exactly the data used in Blocks 13 and 14.
print("'cap' version identical to the Block 13/14 input:", outlier_versions["cap"].equals(X_pipeline_input))


# ----- 15.3 Grid search -----
print("\n--- 15.3 Grid search: 5 pairs x 3 outlier versions x 3 k x 3 model settings, 5-fold CV ---")
# Two scores per setting. Average precision is built with make_scorer and
# response_method="predict_proba", so it ranks wafers by failure probability
# exactly as in Block 14 (the built-in name "average_precision" would use
# decision_function where a model has one). F2 uses the 0/1 predictions at the
# default threshold 0.5; zero_division=0 gives 0 instead of a warning when a
# fold has no predicted failure.
grid_scoring = {"average_precision": make_scorer(average_precision_score, response_method="predict_proba"),
                "F2": make_scorer(fbeta_score, beta=2, zero_division=0)}

grid_rows = []
grid_fold_precisions = {}
grid_start_time = time.perf_counter()
for family, balancing in shortlist_pairs:
    pair_name = family + " + " + balancing
    setting_name, setting_values, setting_default = MODEL_SETTINGS[family]
    # GridSearchCV names a pipeline step's setting "<step name>__<setting>".
    param_grid = {"imputer__n_neighbors": GRID_NEIGHBORS,
                  "model__" + setting_name: setting_values}
    pair_start_time = time.perf_counter()
    for option in GRID_OUTLIER_OPTIONS:
        # GridSearchCV fits a fresh copy of the pipeline for every setting and
        # every fold, so all steps are refitted inside each fold. cv=cv_splitter:
        # the same five parts as before. refit=False: no model on the whole
        # training set here (that is Block 17). n_jobs=None (default): one
        # thread, as for Boruta. error_score="raise": a failing setting stops the
        # script instead of the default silent NaN score.
        search = GridSearchCV(build_pipeline(make_model(family, balancing), balancing),
                              param_grid=param_grid, scoring=grid_scoring, cv=cv_splitter,
                              refit=False, n_jobs=None, error_score="raise")
        search.fit(outlier_versions[option], y_train)
        results = search.cv_results_
        # One row per setting; the standard deviation uses ddof=1 as in Block 14
        # (GridSearchCV's own std_test columns use ddof=0).
        for row_number in range(len(results["params"])):
            settings = results["params"][row_number]
            k_value = settings["imputer__n_neighbors"]
            model_value = settings["model__" + setting_name]
            precisions = fold_scores_from_search(results, row_number, "average_precision")
            f2_values = fold_scores_from_search(results, row_number, "F2")
            grid_fold_precisions[(pair_name, option, k_value, model_value)] = precisions
            grid_rows.append({"pair": pair_name, "family": family, "balancing": balancing,
                              "outlier": option, "k": k_value,
                              "setting": setting_name, "value": model_value,
                              "mean_average_precision": round(float(precisions.mean()), 4),
                              "std_average_precision": round(float(precisions.std(ddof=1)), 4),
                              "mean_F2_at_0.5": round(float(f2_values.mean()), 4)})
    print("  " + pair_name.ljust(34), "27 settings x 5 folds done in",
          round(time.perf_counter() - pair_start_time), "s")
grid_table = pd.DataFrame(grid_rows)
grid_table.to_csv(TABLES_FOLDER / "block15_grid_results.csv", index=False)
print("Grid search finished in", round(time.perf_counter() - grid_start_time), "s;",
      len(grid_table), "settings scored")


# ----- 15.4 Best setting per pair, checks and chart -----
print("\n--- 15.4 Best setting per pair (highest mean average precision) ---")
best_rows = []
tuned_candidates = []
for family, balancing in shortlist_pairs:
    pair_name = family + " + " + balancing
    setting_name, setting_values, setting_default = MODEL_SETTINGS[family]
    pair_rows = grid_table[grid_table["pair"] == pair_name]
    # kind="stable" keeps the grid order for exact ties, so the first setting wins.
    best = pair_rows.sort_values("mean_average_precision", ascending=False, kind="stable").iloc[0]
    best_precisions = grid_fold_precisions[(pair_name, best["outlier"], best["k"], best["value"])]
    default_precisions = grid_fold_precisions[(pair_name, "cap", KNN_NEIGHBORS, setting_default)]
    # Check: the default setting must reproduce Block 14's fold scores exactly.
    matches_block14 = bool(np.allclose(default_precisions, combination_average_precisions[pair_name]))
    # Gain over the default, judged against the spread of the fold differences
    # (the same rule as the D17 voting check).
    gain_per_fold = best_precisions - default_precisions
    # Edge check: a best value at the end of its range may mean the best lies outside it.
    k_at_edge = best["k"] == min(GRID_NEIGHBORS) or best["k"] == max(GRID_NEIGHBORS)
    value_at_edge = best["value"] == setting_values[0] or best["value"] == setting_values[-1]
    best_rows.append({"pair": pair_name, "outlier": best["outlier"], "k": best["k"],
                      "setting": setting_name, "value": best["value"],
                      "tuned_average_precision": best["mean_average_precision"],
                      "tuned_std": best["std_average_precision"],
                      "default_average_precision": round(float(default_precisions.mean()), 4),
                      "mean_gain": round(float(gain_per_fold.mean()), 4),
                      "std_of_gains": round(float(gain_per_fold.std(ddof=1)), 4),
                      "gain_beyond_noise": float(gain_per_fold.mean()) > float(gain_per_fold.std(ddof=1)),
                      "k_at_edge": k_at_edge, "value_at_edge": value_at_edge,
                      "default_matches_block14": matches_block14})
    # The tuned settings of every pair, handed on to Block 16. The "value" column
    # mixes whole numbers and decimals, so pandas stores 10 as 10.0; the value is
    # taken back from the original list, because n_estimators must be a whole number.
    tuned_value = None
    for value in setting_values:
        if value == best["value"]:
            tuned_value = value
    tuned_candidates.append({"family": family, "balancing": balancing, "outlier": best["outlier"],
                             "k": int(best["k"]), "setting": setting_name, "value": tuned_value})
best_settings = pd.DataFrame(best_rows).sort_values("tuned_average_precision", ascending=False)
print(best_settings.to_string(index=False))
best_settings.to_csv(TABLES_FOLDER / "block15_best_settings.csv", index=False)

# Chart: default (Block 14) vs. tuned mean average precision per pair, with
# +/- one standard deviation over the folds.
chart_positions = np.arange(len(best_settings))
figure, axis = plt.subplots(figsize=(10, 5))
default_stds = []
for pair_name in best_settings["pair"]:
    family = pair_name.split(" + ")[0]
    default_key = (pair_name, "cap", KNN_NEIGHBORS, MODEL_SETTINGS[family][2])
    default_stds.append(float(grid_fold_precisions[default_key].std(ddof=1)))
axis.bar(chart_positions - 0.2, best_settings["default_average_precision"], width=0.4,
         yerr=default_stds, capsize=3, color="#a6a6a6", label="default (Block 14)")
axis.bar(chart_positions + 0.2, best_settings["tuned_average_precision"], width=0.4,
         yerr=best_settings["tuned_std"], capsize=3, color=COLOR_MAIN, label="tuned (best setting)")
axis.axhline(float(y_train.mean()), linestyle="--", color=COLOR_REFERENCE,
             label="random guess = failure share (" + str(round(float(y_train.mean()), 3)) + ")")
axis.set_xticks(chart_positions)
axis.set_xticklabels(best_settings["pair"].str.replace(" + ", "\n+ ", regex=False), fontsize=8)
axis.set_ylabel("Average precision, mean over 5 folds")
axis.set_title("Grid search: default vs. tuned settings (training set, 5-fold CV)")
# The y-axis goes up to 0.5 so the legend fits above the error bars (highest about 0.37).
axis.set_ylim(0, 0.5)
axis.legend(fontsize=8, loc="upper right", ncol=3)
axis.spines["top"].set_visible(False)
axis.spines["right"].set_visible(False)
save_figure(figure, "block15_01_default_vs_tuned.png")


# ----- 15.5 Boruta refitted inside every fold, for the best tuned pair -----
best_candidate = None
for candidate in tuned_candidates:
    if candidate["family"] + " + " + candidate["balancing"] == best_settings.iloc[0]["pair"]:
        best_candidate = candidate
best_pair_name = best_candidate["family"] + " + " + best_candidate["balancing"]
print("\n--- 15.5 Boruta inside every fold:", best_pair_name, "with its tuned settings ---")
boruta_start_time = time.perf_counter()
in_fold_precisions, sensor_counts, shared_counts = cross_validate_with_boruta(
    outlier_versions[best_candidate["outlier"]], y_train, best_candidate["family"],
    best_candidate["balancing"], best_candidate["k"], best_candidate["setting"], best_candidate["value"])
fixed_precisions = grid_fold_precisions[(best_pair_name, best_candidate["outlier"],
                                         best_candidate["k"], best_candidate["value"])]
boruta_rows = []
for part in range(CV_FOLDS):
    boruta_rows.append({"fold": part + 1,
                        "average_precision_fixed_18": round(float(fixed_precisions[part]), 4),
                        "average_precision_boruta_in_fold": round(float(in_fold_precisions[part]), 4),
                        "sensors_chosen_in_fold": sensor_counts[part],
                        "of_them_among_fixed_18": shared_counts[part]})
boruta_check = pd.DataFrame(boruta_rows)
print(boruta_check.to_string(index=False))
boruta_check.to_csv(TABLES_FOLDER / "block15_boruta_in_fold_check.csv", index=False)
print("Mean average precision: fixed 18 sensors", round(float(fixed_precisions.mean()), 4),
      "| Boruta inside the folds", round(float(in_fold_precisions.mean()), 4),
      "| optimism of the fixed list", round(float(fixed_precisions.mean() - in_fold_precisions.mean()), 4))
print("Boruta check finished in", round(time.perf_counter() - boruta_start_time), "s")


# %% Block 16: Decision threshold by cost and sensitivity analysis (training set only)
# A missed failure costs 25 times more than a false alarm, so the default rule
# "fail if the probability is above 0.5" is not the cheapest. For each of the
# five tuned pairs (Block 15), this block finds the threshold with the lowest
# total cost on the cross-validated training probabilities (D16; never on the
# test set, RULES_BOOK), chooses the final model by the lowest total cost
# (professor's rule 11; DECISIONS.md, 21 Oct), shows whether the differences
# are within the fold noise, checks how optimistic the threshold is, draws ROC and precision-recall
# curves and repeats the choice for other cost assumptions (manual 14.3).
# Manual Section 14. The test set is not used in this block.


def cost_for_threshold(y_data, probabilities, threshold, cost_fp, cost_fn):
    """
    Total cost when every wafer with a failure probability above the threshold
    is flagged as failed (the same "above" rule as threshold 0.5 before).

    Input:  y_data (Series of true labels), probabilities (Series, same order),
            threshold (float), cost_fp and cost_fn (euros per false positive and
            per false negative)
    Output: total cost (number), false positives (int), false negatives (int)
    """
    predictions = (probabilities > threshold).astype(int)
    false_positives = int(((predictions == 1) & (y_data == 0)).sum())
    false_negatives = int(((predictions == 0) & (y_data == 1)).sum())
    total_cost = cost_fp * false_positives + cost_fn * false_negatives
    return total_cost, false_positives, false_negatives


def best_threshold(y_data, probabilities, cost_fp, cost_fn):
    """
    The threshold from THRESHOLDS with the lowest total cost. On a tie the
    lowest threshold is kept, because a missed failure is the expensive error.

    Input:  y_data (Series), probabilities (Series), cost_fp and cost_fn (euros)
    Output: best threshold (float), its total cost (number)
    """
    best_value = None
    best_cost = None
    for threshold in THRESHOLDS:
        cost, false_positives, false_negatives = cost_for_threshold(y_data, probabilities, threshold, cost_fp, cost_fn)
        # Strictly cheaper only, so the first (lowest) threshold wins a tie.
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_value = threshold
    return best_value, best_cost


def fold_costs_at_threshold(y_data, probabilities, threshold):
    """
    Cost in each of the five CV parts at one threshold (base costs), from the
    out-of-fold probabilities.

    Input:  y_data (Series), probabilities (Series, out-of-fold), threshold (float)
    Output: array with the cost of each part
    """
    costs = []
    for train_positions, validation_positions in cv_splitter.split(y_data, y_data):
        cost, false_positives, false_negatives = cost_for_threshold(
            y_data.iloc[validation_positions], probabilities.iloc[validation_positions],
            threshold, COST_FALSE_POSITIVE, COST_FALSE_NEGATIVE)
        costs.append(cost)
    return np.array(costs)


# ----- 16.1 Out-of-fold probabilities of the five tuned pairs -----
print("\n--- 16.1 Tuned pairs: out-of-fold probabilities (same 5 folds) ---")
# Thresholds 0.01, 0.02, ..., 0.99: np.arange(1, 100) gives 1 to 99, divided
# by 100. A finer step would only fit the noise of 83 failed wafers.
THRESHOLDS = np.round(np.arange(1, 100) / 100, 2)
# Order from simplest to most complex model: the order of all tables and
# figures, and the winner of an exact cost tie (DECISIONS.md, 21 Oct).
SIMPLICITY_ORDER = ["logistic_regression", "random_forest", "balanced_random_forest",
                    "gradient_boosting", "easy_ensemble"]
# One colour per model, used in every Block 16 figure.
CANDIDATE_COLORS = {"logistic_regression": "#2a78d6", "random_forest": "#eb6834",
                    "balanced_random_forest": "#008300", "gradient_boosting": "#8e44ad",
                    "easy_ensemble": "#52514e"}

candidate_names = {}
candidate_probabilities = {}
candidate_average_precisions = {}
for candidate in tuned_candidates:
    family = candidate["family"]
    name = family + " + " + candidate["balancing"]
    candidate_names[family] = name
    # The tuned model: the Block 14 model with the one setting from Block 15.
    model = make_model(family, candidate["balancing"])
    model.set_params(**{candidate["setting"]: candidate["value"]})
    pipeline = build_pipeline(model, candidate["balancing"], n_neighbors=candidate["k"])
    costs_per_fold, probabilities = cross_validate_pipeline(pipeline, outlier_versions[candidate["outlier"]], y_train)
    candidate_probabilities[family] = probabilities
    # Check: the same average precision per fold as in the Block 15 grid search.
    precisions_per_fold, recalls_per_fold = fold_values(y_train, probabilities)
    candidate_average_precisions[family] = float(precisions_per_fold.mean())
    block15_key = (name, candidate["outlier"], candidate["k"], candidate["value"])
    print("  " + name.ljust(32), "AP", round(float(precisions_per_fold.mean()), 4),
          "| matches Block 15:", bool(np.allclose(precisions_per_fold, grid_fold_precisions[block15_key])),
          "| probabilities from", round(float(probabilities.min()), 3), "to", round(float(probabilities.max()), 3))


# ----- 16.2 Total cost against the threshold -----
print("\n--- 16.2 Cost-optimal threshold per model (base costs EUR",
      COST_FALSE_POSITIVE, "/", COST_FALSE_NEGATIVE, ") ---")
curve_rows = []
best_rows = []
candidate_fold_costs = {}
candidate_thresholds = {}
for family in SIMPLICITY_ORDER:
    name = candidate_names[family]
    probabilities = candidate_probabilities[family]
    # Cost at every threshold (the curve), then the cheapest one.
    curve_costs = []
    for threshold in THRESHOLDS:
        cost, false_positives, false_negatives = cost_for_threshold(
            y_train, probabilities, threshold, COST_FALSE_POSITIVE, COST_FALSE_NEGATIVE)
        curve_costs.append(cost)
        curve_rows.append({"model": name, "threshold": threshold, "FP": false_positives,
                           "FN": false_negatives, "cost_eur": cost})
    threshold, lowest_cost = best_threshold(y_train, probabilities, COST_FALSE_POSITIVE, COST_FALSE_NEGATIVE)
    candidate_thresholds[family] = threshold
    candidate_fold_costs[family] = fold_costs_at_threshold(y_train, probabilities, threshold)
    # The "flat valley": all thresholds within 5 % of the lowest cost.
    near_best = []
    for position in range(len(THRESHOLDS)):
        if curve_costs[position] <= 1.05 * lowest_cost:
            near_best.append(THRESHOLDS[position])
    result = full_evaluation(y_train, (probabilities > threshold).astype(int), probabilities)
    cost_at_default, fp_at_default, fn_at_default = cost_for_threshold(
        y_train, probabilities, 0.5, COST_FALSE_POSITIVE, COST_FALSE_NEGATIVE)
    best_rows.append({"model": name, "family": family, "best_threshold": threshold,
                      "cost_eur": result["cost_eur"], "TP": result["TP"], "FP": result["FP"],
                      "FN": result["FN"], "recall": result["recall"],
                      "specificity": result["specificity"], "precision": result["precision"],
                      "F2": result["F2"], "ROC_AUC": result["ROC_AUC"],
                      "within_5_percent_from": min(near_best), "within_5_percent_to": max(near_best),
                      "cost_at_0.5": cost_at_default,
                      "std_cost_per_fold": round(float(candidate_fold_costs[family].std(ddof=1)), 1)})
cost_curves = pd.DataFrame(curve_rows)
cost_curves.to_csv(TABLES_FOLDER / "block16_cost_by_threshold.csv", index=False)
best_thresholds = pd.DataFrame(best_rows).sort_values("cost_eur", kind="stable")
print(best_thresholds.drop(columns="family").to_string(index=False))
best_thresholds.to_csv(TABLES_FOLDER / "block16_best_thresholds.csv", index=False)

# Figure: one cost curve per model, its minimum marked, threshold 0.5 dashed.
figure, axis = plt.subplots(figsize=(10, 6))
for family in SIMPLICITY_ORDER:
    name = candidate_names[family]
    model_curve = cost_curves[cost_curves["model"] == name]
    axis.plot(model_curve["threshold"], model_curve["cost_eur"] / 1000, color=CANDIDATE_COLORS[family],
              linewidth=1.8, label=name + " (best " + str(candidate_thresholds[family]) + ")")
    lowest_row = best_thresholds[best_thresholds["family"] == family].iloc[0]
    axis.plot(lowest_row["best_threshold"], lowest_row["cost_eur"] / 1000, marker="o", markersize=8,
              color=CANDIDATE_COLORS[family])
axis.axvline(0.5, linestyle="--", color=COLOR_REFERENCE)
axis.text(0.505, axis.get_ylim()[1] * 0.98, "default 0.5", color=COLOR_REFERENCE, va="top", fontsize=8)
axis.set_xlabel("Decision threshold (wafer flagged as failed if its probability is above it)")
axis.set_ylabel("Total cost (thousand EUR): 200 x FP + 5,000 x FN")
axis.set_title("Total cost against the decision threshold (training set, cross-validated probabilities)")
axis.legend(fontsize=8)
axis.spines["top"].set_visible(False)
axis.spines["right"].set_visible(False)
save_figure(figure, "block16_01_cost_vs_threshold.png")


# ----- 16.3 Final model: the lowest total cost (rule 11) -----
print("\n--- 16.3 Final model choice: lowest total cost at its own threshold ---")
# best_thresholds is sorted by cost with kind="stable" from a table built in
# SIMPLICITY_ORDER, so on an exact tie the simpler model comes first.
final_family = best_thresholds.iloc[0]["family"]
# For information only (not used for the choice): is each model's extra cost
# per fold larger than the fold-to-fold noise? A tie rule based on this test
# chose the most expensive model in the first run, because with 83 failed
# wafers the noise is large (DECISIONS.md, 21 Oct).
choice_rows = []
for family in SIMPLICITY_ORDER:
    extra_per_fold = candidate_fold_costs[family] - candidate_fold_costs[final_family]
    clearly_worse = float(extra_per_fold.mean()) > float(extra_per_fold.std(ddof=1))
    if family == final_family:
        clearly_worse = False
    choice_rows.append({"model": candidate_names[family],
                        "mean_extra_cost_per_fold": round(float(extra_per_fold.mean()), 1),
                        "std_of_extra_cost": round(float(extra_per_fold.std(ddof=1)), 1),
                        "clearly_worse_than_final": clearly_worse})
choice_table = pd.DataFrame(choice_rows)
print(choice_table.to_string(index=False))
choice_table.to_csv(TABLES_FOLDER / "block16_final_choice.csv", index=False)
print("Chosen (lowest cost):", candidate_names[final_family])

final_probabilities = candidate_probabilities[final_family]
final_threshold = candidate_thresholds[final_family]
final_predictions = (final_probabilities > final_threshold).astype(int)
plot_confusion_matrix(y_train, final_predictions,
                      "Final model " + candidate_names[final_family] + ", threshold " + str(final_threshold)
                      + "\n(training set, cross-validated)", "block16_04_confusion_matrix_final.png")


# ----- 16.4 How optimistic is a threshold chosen on the same predictions? -----
print("\n--- 16.4 Threshold check: learned on 4 folds, applied to the 5th ---")
check_rows = []
fold_number = 0
for train_positions, validation_positions in cv_splitter.split(y_train, y_train):
    fold_number = fold_number + 1
    # The threshold is learned only from the other four parts' probabilities.
    learned_threshold, learned_cost = best_threshold(y_train.iloc[train_positions],
                                                     final_probabilities.iloc[train_positions],
                                                     COST_FALSE_POSITIVE, COST_FALSE_NEGATIVE)
    cost_learned, fp_learned, fn_learned = cost_for_threshold(
        y_train.iloc[validation_positions], final_probabilities.iloc[validation_positions],
        learned_threshold, COST_FALSE_POSITIVE, COST_FALSE_NEGATIVE)
    cost_pooled, fp_pooled, fn_pooled = cost_for_threshold(
        y_train.iloc[validation_positions], final_probabilities.iloc[validation_positions],
        final_threshold, COST_FALSE_POSITIVE, COST_FALSE_NEGATIVE)
    check_rows.append({"fold": fold_number, "threshold_from_other_folds": learned_threshold,
                       "cost_with_that_threshold": cost_learned,
                       "cost_with_pooled_threshold": cost_pooled})
threshold_check = pd.DataFrame(check_rows)
print(threshold_check.to_string(index=False))
threshold_check.to_csv(TABLES_FOLDER / "block16_threshold_check.csv", index=False)
print("Total: pooled threshold", int(threshold_check["cost_with_pooled_threshold"].sum()),
      "EUR | threshold learned without the fold", int(threshold_check["cost_with_that_threshold"].sum()), "EUR")


# ----- 16.5 ROC and precision-recall curves of the five tuned pairs -----
# Each curve shows every possible threshold at once; the dot is the model's
# cost-optimal threshold from 16.2.
figure_roc, axis_roc = plt.subplots(figsize=(7, 7))
figure_pr, axis_pr = plt.subplots(figsize=(7, 7))
for family in SIMPLICITY_ORDER:
    name = candidate_names[family]
    probabilities = candidate_probabilities[family]
    lowest_row = best_thresholds[best_thresholds["family"] == family].iloc[0]
    false_positive_rate, true_positive_rate, roc_thresholds = roc_curve(y_train, probabilities)
    axis_roc.plot(false_positive_rate, true_positive_rate, color=CANDIDATE_COLORS[family], linewidth=1.8,
                  label=name + " (AUC " + str(round(float(roc_auc_score(y_train, probabilities)), 3)) + ")")
    axis_roc.plot(1 - lowest_row["specificity"], lowest_row["recall"], marker="o", markersize=8,
                  color=CANDIDATE_COLORS[family])
    # precision_recall_curve returns precision and recall for every threshold.
    precision_values, recall_values, pr_thresholds = precision_recall_curve(y_train, probabilities)
    axis_pr.plot(recall_values, precision_values, color=CANDIDATE_COLORS[family], linewidth=1.8,
                 label=name + " (AP " + str(round(float(average_precision_score(y_train, probabilities)), 3)) + ")")
    axis_pr.plot(lowest_row["recall"], lowest_row["precision"], marker="o", markersize=8,
                 color=CANDIDATE_COLORS[family])
axis_roc.plot([0, 1], [0, 1], linestyle="--", color=COLOR_REFERENCE, label="random guess (AUC 0.5)")
axis_roc.set_xlabel("False-positive rate (good wafers flagged)")
axis_roc.set_ylabel("True-positive rate = recall (failed wafers caught)")
axis_roc.set_title("ROC curves, tuned pairs (training set, cross-validated)\ndots: cost-optimal thresholds", fontsize=10)
axis_roc.legend(loc="lower right", fontsize=8)
axis_roc.spines["top"].set_visible(False)
axis_roc.spines["right"].set_visible(False)
save_figure(figure_roc, "block16_02_roc_curves.png")
axis_pr.axhline(float(y_train.mean()), linestyle="--", color=COLOR_REFERENCE,
                label="random guess = failure share (" + str(round(float(y_train.mean()), 3)) + ")")
axis_pr.set_xlabel("Recall (failed wafers caught)")
axis_pr.set_ylabel("Precision (flagged wafers that really failed)")
axis_pr.set_title("Precision-recall curves, tuned pairs (training set, cross-validated)\n"
                  "dots: cost-optimal thresholds", fontsize=10)
axis_pr.legend(loc="upper right", fontsize=8)
axis_pr.spines["top"].set_visible(False)
axis_pr.spines["right"].set_visible(False)
save_figure(figure_pr, "block16_03_precision_recall_curves.png")


# ----- 16.6 Sensitivity analysis: other cost assumptions (manual 14.3) -----
print("\n--- 16.6 Sensitivity analysis: best threshold and model under other costs ---")
# Each cost changed by -50, -20, +20 and +50 % while the other stays fixed,
# plus the manual's example of EUR 100 / 400. Only the probabilities from 16.1
# are reused; no model is retrained.
cost_scenarios = [("baseline", 200, 5000),
                  ("FP -50 %", 100, 5000), ("FP -20 %", 160, 5000),
                  ("FP +20 %", 240, 5000), ("FP +50 %", 300, 5000),
                  ("FN -50 %", 200, 2500), ("FN -20 %", 200, 4000),
                  ("FN +20 %", 200, 6000), ("FN +50 %", 200, 7500),
                  ("manual example", 100, 400)]
sensitivity_rows = []
summary_rows = []
for scenario_name, cost_fp, cost_fn in cost_scenarios:
    scenario_costs = {}
    for family in SIMPLICITY_ORDER:
        probabilities = candidate_probabilities[family]
        threshold, cost = best_threshold(y_train, probabilities, cost_fp, cost_fn)
        cost_again, false_positives, false_negatives = cost_for_threshold(y_train, probabilities, threshold, cost_fp, cost_fn)
        scenario_costs[family] = cost
        row = {"scenario": scenario_name, "cost_FP": cost_fp, "cost_FN": cost_fn,
               "ratio_FN_to_FP": round(cost_fn / cost_fp, 2), "model": candidate_names[family],
               "best_threshold": threshold, "cost_eur": cost, "FP": false_positives,
               "FN": false_negatives, "recall": round(1 - false_negatives / int(y_train.sum()), 4)}
        sensitivity_rows.append(row)
        if family == final_family:
            final_model_row = row
    # Cheapest model in this scenario (first in simplicity order on a tie), and
    # the final model's rank: 1 + the number of models that are cheaper.
    cheapest_in_scenario = SIMPLICITY_ORDER[0]
    final_model_rank = 1
    for family in SIMPLICITY_ORDER:
        if scenario_costs[family] < scenario_costs[cheapest_in_scenario]:
            cheapest_in_scenario = family
        if scenario_costs[family] < scenario_costs[final_family]:
            final_model_rank = final_model_rank + 1
    summary_rows.append({"scenario": scenario_name, "ratio_FN_to_FP": round(cost_fn / cost_fp, 2),
                         "cheapest_model": candidate_names[cheapest_in_scenario],
                         "final_model_threshold": final_model_row["best_threshold"],
                         "final_model_cost": final_model_row["cost_eur"],
                         "final_model_recall": final_model_row["recall"],
                         "final_model_rank": final_model_rank})
sensitivity_table = pd.DataFrame(sensitivity_rows)
sensitivity_table.to_csv(TABLES_FOLDER / "block16_sensitivity.csv", index=False)
sensitivity_summary = pd.DataFrame(summary_rows)
print(sensitivity_summary.to_string(index=False))
sensitivity_summary.to_csv(TABLES_FOLDER / "block16_sensitivity_summary.csv", index=False)

# Figure: best threshold against the cost ratio (log scale), one line per model.
figure, axis = plt.subplots(figsize=(9, 5.5))
for family in SIMPLICITY_ORDER:
    model_rows = sensitivity_table[sensitivity_table["model"] == candidate_names[family]]
    model_rows = model_rows.sort_values("ratio_FN_to_FP")
    axis.plot(model_rows["ratio_FN_to_FP"], model_rows["best_threshold"], marker="o",
              color=CANDIDATE_COLORS[family], label=candidate_names[family])
axis.axvline(COST_FALSE_NEGATIVE / COST_FALSE_POSITIVE, linestyle="--", color=COLOR_REFERENCE)
axis.text(COST_FALSE_NEGATIVE / COST_FALSE_POSITIVE * 1.03, 0.97, "base 5,000 / 200 = 25",
          color=COLOR_REFERENCE, fontsize=8, va="top")
axis.set_xscale("log")
# Plain numbers at the scenario ratios instead of the default "4 x 10^0" labels.
axis.set_xticks([4, 12.5, 20, 25, 30, 37.5, 50])
axis.set_xticklabels(["4", "12.5", "20", "25", "30", "37.5", "50"])
axis.minorticks_off()
axis.set_ylim(0, 1)
axis.set_xlabel("Cost ratio: false negative / false positive (log scale)")
axis.set_ylabel("Cost-optimal threshold")
axis.set_title("Sensitivity analysis: how the best threshold moves with the cost ratio (training set)")
axis.legend(fontsize=8)
axis.spines["top"].set_visible(False)
axis.spines["right"].set_visible(False)
save_figure(figure, "block16_05_sensitivity.png")


# ----- 16.7 The final decisions, stored for Block 17 -----
print("\n--- 16.7 Final decisions (fixed from here on; Block 17 only applies them) ---")
for candidate in tuned_candidates:
    if candidate["family"] == final_family:
        final_choice = dict(candidate)
final_choice["threshold"] = final_threshold
for key in final_choice:
    print("  " + key.ljust(10), final_choice[key])


# ----- 16.8 Decision figures for the presentation -----
# Seven figures that explain the final choice step by step (why a threshold by
# cost, why EasyEnsemble, why 0.49, how robust, what it means for inspection,
# how the selection narrowed down). They only reuse the training-set results of
# Blocks 14 to 16 (cross-validated); nothing is refitted and the test set is
# not used. The final model is blue, all other models grey.
print("\n--- 16.8 Decision figures for the presentation ---")
FINAL_COLOR = COLOR_MAIN
OTHER_COLOR = "#b0b0b0"
COLOR_MISSED = COLOR_FAIL
COLOR_ALARMS = "#9ecae1"
# Readable model names for slides instead of the code names.
FAMILY_LABELS = {"logistic_regression": "Logistic regression", "random_forest": "Random forest",
                 "balanced_random_forest": "Balanced random forest",
                 "gradient_boosting": "Gradient boosting", "easy_ensemble": "EasyEnsemble"}
# Readable names of the tuned model settings (Block 15).
SETTING_LABELS = {"n_estimators": "sub-models:", "min_samples_leaf": "leaf size",
                  "learning_rate": "learning rate", "C": "C ="}


def display_name(family):
    """
    Readable name of a tuned model for the figures, for example
    "Random forest + ROSE" or "EasyEnsemble" (no "+ none").

    Input:  family (str, a key of candidate_names)
    Output: the name (str)
    """
    balancing = candidate_names[family].split(" + ")[1]
    if balancing == "none":
        return FAMILY_LABELS[family]
    return FAMILY_LABELS[family] + " + " + balancing


def plain_number(value):
    """
    A number as short text: 30.0 becomes "30", 12.5 stays "12.5".

    Input:  value (int or float)
    Output: str
    """
    if float(value) == int(value):
        return str(int(value))
    return str(value)


def presentation_axis(axis, title, x_label, y_label):
    """
    Give an axis the common style of the decision figures: a title that states
    the message, readable labels for slides, no top and right frame lines.

    Input:  axis (matplotlib Axes), title, x_label, y_label (str)
    Output: none (the axis is changed in place)
    """
    axis.set_title(title, fontsize=12)
    axis.set_xlabel(x_label, fontsize=11)
    axis.set_ylabel(y_label, fontsize=11)
    axis.tick_params(labelsize=10)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


number_of_wafers = len(y_train)
number_of_failures = int(y_train.sum())
number_of_passes = number_of_wafers - number_of_failures
final_label = display_name(final_family)
print("Final model mean average precision (6 decimals):", round(candidate_average_precisions[final_family], 6))
# The final model's rows of the cost-by-threshold table (16.2), with the number
# of failures caught and the share of wafers inspected (flagged) per threshold.
final_curve = cost_curves[cost_curves["model"] == candidate_names[final_family]].copy()
final_curve["caught"] = number_of_failures - final_curve["FN"]
final_curve["inspected_percent"] = (final_curve["FP"] + final_curve["caught"]) / number_of_wafers * 100
final_curve["caught_percent"] = final_curve["caught"] / number_of_failures * 100
# np.isclose avoids problems with tiny rounding differences in decimal thresholds.
final_row = final_curve[np.isclose(final_curve["threshold"], final_threshold)].iloc[0]
default_row = final_curve[np.isclose(final_curve["threshold"], 0.5)].iloc[0]


# Figure 1: where the money goes, from "no screening" to the final threshold.
strategies = [("Flag no wafer", 0, number_of_failures),
              ("Flag every wafer", number_of_passes, 0),
              (final_label + "\nat 0.5", int(default_row["FP"]), int(default_row["FN"])),
              (final_label + "\nat " + str(final_threshold), int(final_row["FP"]), int(final_row["FN"]))]
figure, axis = plt.subplots(figsize=(10, 6))
strategy_labels = []
for position in range(len(strategies)):
    label, false_alarms, missed = strategies[position]
    strategy_labels.append(label)
    missed_cost = missed * COST_FALSE_NEGATIVE / 1000
    alarm_cost = false_alarms * COST_FALSE_POSITIVE / 1000
    # Stacked bar: missed failures at the bottom, false alarms on top; the
    # legend label is given only once (first bar).
    missed_label = None
    alarm_label = None
    if position == 0:
        missed_label = "missed failures (EUR 5,000 each)"
        alarm_label = "false alarms (EUR 200 each)"
    axis.bar(position, missed_cost, color=COLOR_MISSED, width=0.6, label=missed_label)
    axis.bar(position, alarm_cost, bottom=missed_cost, color=COLOR_ALARMS, width=0.6, label=alarm_label)
    axis.text(position, missed_cost + alarm_cost + 6, "EUR " + format(int((missed_cost + alarm_cost) * 1000), ","),
              ha="center", fontsize=11, fontweight="bold")
    axis.text(position, -32, str(number_of_failures - missed) + " of " + str(number_of_failures) + " caught\n"
              + format(false_alarms, ",") + " false alarms", ha="center", va="top", fontsize=9)
axis.set_xticks(range(len(strategies)))
axis.set_xticklabels(strategy_labels, fontsize=10)
axis.set_ylim(-75, 470)
axis.axhline(0, color="#595959", linewidth=0.8)
axis.legend(fontsize=9, loc="upper right")
saving_vs_default = int(default_row["cost_eur"] - final_row["cost_eur"])
presentation_axis(axis, "Setting the threshold by cost saves EUR " + format(saving_vs_default, ",")
                  + " against the default 0.5\n(training set, cross-validated, " + format(number_of_wafers, ",")
                  + " wafers)", "", "Total cost (thousand EUR)")
save_figure(figure, "block16_06_decision_cost_breakdown.png")

# Figure 2: the five tuned models on cost, average precision and ROC-AUC.
# Rows from the most expensive (bottom) to the cheapest (top).
comparison = best_thresholds.sort_values("cost_eur", ascending=False, kind="stable")
row_labels = []
bar_colors = []
cost_values = []
precision_values = []
auc_values = []
for row_number in range(len(comparison)):
    family = comparison.iloc[row_number]["family"]
    row_labels.append(display_name(family))
    bar_colors.append(FINAL_COLOR if family == final_family else OTHER_COLOR)
    cost_values.append(float(comparison.iloc[row_number]["cost_eur"]) / 1000)
    # Mean average precision over the 5 folds (the selection score), unrounded.
    precision_values.append(candidate_average_precisions[family])
    auc_values.append(float(comparison.iloc[row_number]["ROC_AUC"]))
figure, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
panels = [(cost_values, "Total cost at own best threshold t\n(thousand EUR, lower is better)", None),
          (precision_values, "Average precision, mean of 5 folds\n(higher is better)", float(y_train.mean())),
          (auc_values, "ROC-AUC\n(higher is better)", 0.5)]
top_position = len(comparison) - 0.35
for panel in range(3):
    values, label, guess_line = panels[panel]
    axes[panel].barh(range(len(values)), values, color=bar_colors)
    for row in range(len(values)):
        # Panel 0 shows euros and the model's threshold t; the others 3 decimals.
        if panel == 0:
            text = ("EUR " + format(int(round(values[row] * 1000)), ",") + " (t = "
                    + str(comparison.iloc[row]["best_threshold"]) + ")")
        else:
            text = format(values[row], ".3f")
        axes[panel].text(values[row], row, " " + text, va="center", fontsize=9)
    # The random-guess line, labelled above the top bar so nothing overlaps.
    if guess_line is not None:
        axes[panel].axvline(guess_line, linestyle="--", color=COLOR_REFERENCE)
        axes[panel].text(guess_line, top_position, " random guess", color=COLOR_REFERENCE, fontsize=8)
    axes[panel].set_xlim(0, max(values) * 1.5)
    axes[panel].set_ylim(-0.6, len(values) - 0.1)
    presentation_axis(axes[panel], "", label, "")
axes[0].set_yticks(range(len(comparison)))
axes[0].set_yticklabels(row_labels, fontsize=10)
clearly_worse_families = []
for row_number in range(len(choice_table)):
    if choice_table.iloc[row_number]["clearly_worse_than_final"]:
        for family in SIMPLICITY_ORDER:
            if candidate_names[family] == choice_table.iloc[row_number]["model"]:
                clearly_worse_families.append(display_name(family))
figure.suptitle(final_label + " is best on all three measures (training set, cross-validated)", fontsize=13)
figure.text(0.5, -0.1, "Note: with 83 failed wafers the fold-to-fold noise is large; only "
            + " and ".join(clearly_worse_families) + " is more expensive than " + final_label
            + " beyond that noise.", ha="center", fontsize=9, color=COLOR_REFERENCE)
save_figure(figure, "block16_07_decision_model_comparison.png")

# Figure 3: failures caught against false alarms - who catches more at equal alarms?
alarm_budget = int(final_row["FP"])
figure, axis = plt.subplots(figsize=(10, 6.5))
caught_at_budget = {}
for family in SIMPLICITY_ORDER:
    model_curve = cost_curves[cost_curves["model"] == candidate_names[family]].sort_values("FP")
    caught = number_of_failures - model_curve["FN"]
    is_final = family == final_family
    axis.step(model_curve["FP"], caught, where="post", color=FINAL_COLOR if is_final else OTHER_COLOR,
              linewidth=3 if is_final else 1.5, zorder=3 if is_final else 2)
    # Failures caught with at most the final model's number of false alarms,
    # marked as a dot on the dashed line.
    within_budget = model_curve[model_curve["FP"] <= alarm_budget]
    caught_at_budget[family] = int(number_of_failures - within_budget["FN"].min())
    axis.plot(alarm_budget, caught_at_budget[family], marker="o", markersize=8 if is_final else 6,
              color=FINAL_COLOR if is_final else "#7f7f7f", zorder=4)
axis.axvline(alarm_budget, linestyle="--", color=COLOR_REFERENCE)
# A small list in the empty lower-right corner, highest first; sort(reverse=True)
# orders the (caught, family) pairs by the number caught.
label_rows = []
for family in SIMPLICITY_ORDER:
    label_rows.append((caught_at_budget[family], family))
label_rows.sort(reverse=True)
axis.text(0.55, 0.40, "Failures caught with " + str(alarm_budget) + " false alarms:", transform=axis.transAxes,
          fontsize=10, fontweight="bold")
for position in range(len(label_rows)):
    caught, family = label_rows[position]
    is_final = family == final_family
    axis.text(0.57, 0.34 - position * 0.055, str(caught) + "   " + display_name(family), transform=axis.transAxes,
              fontsize=10, color=FINAL_COLOR if is_final else "#595959",
              fontweight="bold" if is_final else "normal")
second_best = label_rows[1][0]
axis.set_xlim(0, number_of_passes)
axis.set_ylim(0, number_of_failures + 5)
presentation_axis(axis, "With " + str(alarm_budget) + " false alarms, " + final_label + " catches "
                  + str(caught_at_budget[final_family]) + " failures; the next best catches " + str(second_best)
                  + "\n(training set, cross-validated; each line runs through thresholds 0.99 to 0.01)",
                  "False alarms (good wafers flagged, of " + format(number_of_passes, ",") + ")",
                  "Failures caught (of " + str(number_of_failures) + ")")
save_figure(figure, "block16_08_decision_same_alarms.png")

# Figure 4: why 0.49 - the two kinds of cost and their sum around the minimum.
zoom = final_curve[(final_curve["threshold"] >= 0.40) & (final_curve["threshold"] <= 0.60)]
figure, axis = plt.subplots(figsize=(10, 6))
axis.plot(zoom["threshold"], zoom["FP"] * COST_FALSE_POSITIVE / 1000, color=COLOR_ALARMS, linewidth=2.5,
          label="cost of false alarms")
axis.plot(zoom["threshold"], zoom["FN"] * COST_FALSE_NEGATIVE / 1000, color=COLOR_MISSED, linewidth=2.5,
          label="cost of missed failures")
axis.plot(zoom["threshold"], zoom["cost_eur"] / 1000, color=FINAL_COLOR, linewidth=3.5, label="total cost")
axis.plot(final_threshold, final_row["cost_eur"] / 1000, marker="o", markersize=11, color=FINAL_COLOR)
axis.annotate("lowest total cost: " + str(final_threshold) + "\nEUR " + format(int(final_row["cost_eur"]), ",")
              + ", " + str(int(final_row["caught"])) + " of " + str(number_of_failures) + " caught",
              xy=(final_threshold, final_row["cost_eur"] / 1000), xytext=(final_threshold - 0.085, 60),
              fontsize=10, arrowprops={"arrowstyle": "->", "color": FINAL_COLOR})
axis.axvline(0.5, linestyle="--", color=COLOR_REFERENCE)
axis.text(0.502, 395, "default 0.5:\nEUR " + format(int(default_row["cost_eur"]), ","), fontsize=9, color=COLOR_REFERENCE)
# Share of wafers inspected at a few thresholds, written above the curves.
for threshold in [0.42, 0.45, final_threshold, 0.52, 0.55, 0.58]:
    row = zoom[np.isclose(zoom["threshold"], threshold)]
    if len(row) > 0:
        axis.text(threshold, 440, str(int(round(float(row["inspected_percent"].iloc[0])))) + " %",
                  ha="center", fontsize=9, color="#595959")
axis.text(0.40, 462, "wafers inspected:", fontsize=9, color="#595959")
axis.set_ylim(0, 480)
axis.legend(fontsize=9, loc="center left")
presentation_axis(axis, "At " + str(final_threshold) + " the total cost is lowest: below it false alarms grow,"
                  " above it missed failures grow\n(" + final_label + ", training set, cross-validated)",
                  "Decision threshold", "Cost (thousand EUR)")
save_figure(figure, "block16_09_decision_why_threshold.png")

# Figure 5: is the threshold stable? (a) across folds, (b) across cost ratios.
final_sensitivity = sensitivity_table[sensitivity_table["model"] == candidate_names[final_family]]
final_sensitivity = final_sensitivity.sort_values("ratio_FN_to_FP")
same_threshold = final_sensitivity[np.isclose(final_sensitivity["best_threshold"], final_threshold)]
lowest_ratio = plain_number(same_threshold["ratio_FN_to_FP"].min())
highest_ratio = plain_number(same_threshold["ratio_FN_to_FP"].max())
# wspace=0.9 leaves room between the panels for the scenario names of panel (b).
figure, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"wspace": 0.9})
axes[0].plot(threshold_check["fold"], threshold_check["threshold_from_other_folds"], marker="o",
             markersize=11, linestyle="none", color=FINAL_COLOR)
axes[0].axhline(final_threshold, linestyle="--", color=COLOR_REFERENCE)
axes[0].set_ylim(0.3, 0.7)
axes[0].set_xticks(list(threshold_check["fold"]))
presentation_axis(axes[0], "(a) Threshold learned without each fold:\n" + str(final_threshold)
                  + " in every fold", "Fold left out", "Cost-optimal threshold")
# Panel (b): one row per cost scenario (smallest cost ratio at the top), a dot
# at its best threshold; blue if it is the final threshold, grey if not. Only
# dots are drawn, because the scenarios between the tested ratios were not tried.
for row_number in range(len(final_sensitivity)):
    scenario = final_sensitivity.iloc[row_number]
    # Readable scenario name, for example "Missed-failure cost +20 %".
    if scenario["scenario"] == "baseline":
        scenario_text = "Our costs: EUR 200 / 5,000"
    elif scenario["scenario"] == "manual example":
        scenario_text = "Manual example: EUR 100 / 400"
    elif scenario["scenario"].startswith("FP"):
        scenario_text = "False-alarm cost " + scenario["scenario"][3:]
    else:
        scenario_text = "Missed-failure cost " + scenario["scenario"][3:]
    scenario_text = scenario_text + "  (ratio " + plain_number(round(float(scenario["ratio_FN_to_FP"]), 2)) + ")"
    row_position = len(final_sensitivity) - 1 - row_number
    is_final_threshold = bool(np.isclose(scenario["best_threshold"], final_threshold))
    axes[1].plot(scenario["best_threshold"], row_position, marker="o", markersize=10,
                 color=FINAL_COLOR if is_final_threshold else OTHER_COLOR)
    # The value is written on the side away from the dashed line, so they never cross.
    if scenario["best_threshold"] < final_threshold:
        axes[1].text(scenario["best_threshold"] - 0.012, row_position, str(scenario["best_threshold"]),
                     ha="right", va="center", fontsize=9, color="#595959")
    else:
        axes[1].text(scenario["best_threshold"] + 0.012, row_position, str(scenario["best_threshold"]),
                     va="center", fontsize=9, color="#595959")
    axes[1].text(0.295, row_position, scenario_text, ha="right", va="center", fontsize=9,
                 fontweight="bold" if scenario["scenario"] == "baseline" else "normal")
axes[1].axvline(final_threshold, linestyle="--", color=COLOR_REFERENCE)
axes[1].set_xlim(0.3, 0.7)
axes[1].set_ylim(-0.7, len(final_sensitivity) - 0.3)
axes[1].set_yticks([])
presentation_axis(axes[1], "(b) Best threshold under 10 cost assumptions:\n" + str(final_threshold) + " in "
                  + str(len(same_threshold)) + " of " + str(len(final_sensitivity)) + " (every ratio from "
                  + lowest_ratio + " to " + highest_ratio + ")", "Cost-optimal threshold", "")
# y=1.05 places the main title above the two panel titles.
figure.suptitle("The threshold " + str(final_threshold) + " is stable (" + final_label
                + ", training set, cross-validated)", fontsize=13, y=1.05)
save_figure(figure, "block16_10_decision_robustness.png")

# Figure 6: what it means for the factory - share inspected against share caught.
figure, axis = plt.subplots(figsize=(9, 7))
workload = final_curve.sort_values("inspected_percent")
axis.plot([0] + list(workload["inspected_percent"]) + [100], [0] + list(workload["caught_percent"]) + [100],
          color=FINAL_COLOR, linewidth=3, label=final_label)
axis.plot([0, 100], [0, 100], linestyle="--", color=COLOR_REFERENCE, label="random checking")
axis.plot(final_row["inspected_percent"], final_row["caught_percent"], marker="o", markersize=12, color=FINAL_COLOR)
axis.annotate("threshold " + str(final_threshold) + ": inspect " + str(int(round(final_row["inspected_percent"])))
              + " %,\ncatch " + str(int(round(final_row["caught_percent"]))) + " % of failures",
              xy=(final_row["inspected_percent"], final_row["caught_percent"]),
              xytext=(final_row["inspected_percent"] + 8, final_row["caught_percent"] - 22),
              fontsize=10, arrowprops={"arrowstyle": "->", "color": FINAL_COLOR})
# Possible inspection capacities: the threshold that catches the most failures
# while staying within the limit (on a tie, the cheaper one). Labels: the 20 %
# one to the left of its line (above the curve), the 30 % one to the right
# (below the curve), so they do not overlap.
capacity_label_places = {20: (-0.8, 86, "right", "top"), 30: (0.8, 4, "left", "bottom")}
for capacity in [20, 30]:
    allowed = final_curve[final_curve["inspected_percent"] <= capacity]
    best_allowed = allowed.sort_values(["caught", "cost_eur"], ascending=[False, True], kind="stable").iloc[0]
    x_shift, y_place, alignment, vertical = capacity_label_places[capacity]
    axis.axvline(capacity, linestyle=":", color="#7f7f7f")
    axis.text(capacity + x_shift, y_place, "capacity " + str(capacity) + " %:\nthreshold "
              + str(best_allowed["threshold"]) + ", catch\n" + str(int(best_allowed["caught"])) + " of "
              + str(number_of_failures) + " (" + str(int(round(best_allowed["caught_percent"]))) + " %)",
              ha=alignment, va=vertical, fontsize=9, color="#595959")
axis.set_xlim(0, 100)
axis.set_ylim(0, 102)
axis.legend(fontsize=9, loc="lower right")
presentation_axis(axis, "Inspecting " + str(int(round(final_row["inspected_percent"]))) + " % of wafers catches "
                  + str(int(round(final_row["caught_percent"]))) + " % of failures\n(training set, cross-validated)",
                  "Wafers inspected (%)", "Failed wafers caught (%)")
save_figure(figure, "block16_11_decision_workload.png")

# Figure 7: the selection funnel from Block 14 to the final model.
funnel_steps = [str(len(combinations)) + " model x balancing combinations compared (Block 14)",
                str(len(shortlist_pairs)) + " model families shortlisted by average precision (Block 14)",
                str(len(grid_table)) + " tuned settings: outliers x k x one model setting (Block 15)",
                str(len(best_thresholds)) + " tuned models, each at its cost-optimal threshold (Block 16)",
                "1 final model: " + final_label + " (" + SETTING_LABELS[final_choice["setting"]] + " "
                + str(final_choice["value"]) + "),\nthreshold " + str(final_threshold) + ", cost EUR " + format(int(final_row["cost_eur"]), ",")]
funnel_widths = [1.0, 0.86, 0.72, 0.58, 0.58]
funnel_colors = ["#e3e3e3", "#d0d0d0", "#bdbdbd", "#a9a9a9", FINAL_COLOR]
figure, axis = plt.subplots(figsize=(11, 6))
for step in range(len(funnel_steps)):
    height_position = len(funnel_steps) - step
    # A centred bar: it starts at -width/2, so all boxes share the same middle.
    axis.barh(height_position, funnel_widths[step], left=-funnel_widths[step] / 2, height=0.8,
              color=funnel_colors[step])
    axis.text(0, height_position, funnel_steps[step], ha="center", va="center", fontsize=10,
              color="white" if step == len(funnel_steps) - 1 else "#222222")
axis.set_xlim(-0.55, 0.55)
axis.axis("off")
axis.set_title("How the final model was selected (training set only, 5-fold CV;\n"
               "the " + str(len(X_test)) + " test wafers stay untouched until Block 17)", fontsize=12)
save_figure(figure, "block16_12_decision_selection_funnel.png")


# %% Block 17: The one and only final test (test set, used once)
# The honest estimate of how the final model performs on new wafers (manual
# Section 15). Every decision was fixed and logged before this block (DECISIONS.md,
# 21 Oct, Step 3.5). The final pipeline is refitted on all 1,253 training wafers;
# the test set only receives the stored rules and transforms (no fitting), is
# predicted at the chosen threshold and is scored once. Nothing may be changed
# because of these numbers (professor's rules 1, 7 and 9).


def wilson_interval(successes, total):
    """
    95 % confidence range for a share (for example recall = caught / failed),
    by the Wilson score method, which stays between 0 and 1 and suits small
    numbers such as 21 failed test wafers.

    Input:  successes (int), total (int)
    Output: lower and upper limit (floats between 0 and 1)
    """
    # z = 1.96: 95 % of a normal distribution lies within 1.96 standard deviations.
    z = 1.96
    share = successes / total
    denominator = 1 + z * z / total
    centre = (share + z * z / (2 * total)) / denominator
    half_width = z * np.sqrt(share * (1 - share) / total + z * z / (4 * total * total)) / denominator
    return centre - half_width, centre + half_width


def count_changed_values(before, after):
    """
    Number of values that differ between two versions of the same table;
    cells missing in both versions are not counted (NaN != NaN in pandas).

    Input:  before, after (DataFrames with the same rows and columns)
    Output: int
    """
    both_missing = before.isna() & after.isna()
    return int(((after != before) & ~both_missing).sum().sum())


# ----- 17.1 The decision sheet: everything fixed before the test -----
print("\n--- 17.1 Final decisions (fixed before the test set is used) ---")
decision_sheet = pd.DataFrame([
    {"step": "Positive class", "decision": "failed wafer = 1, passed wafer = 0"},
    {"step": "Split", "decision": "stratified 80/20, seed " + str(SEED) + ": " + str(len(y_train)) + " training ("
     + str(int(y_train.sum())) + " failed) / " + str(len(y_test)) + " test wafers"},
    {"step": "Column removal", "decision": str(len(columns_removed_in_block5)) + " columns removed, "
     + str(X_train.shape[1]) + " -> " + str(X_train_reduced.shape[1])},
    {"step": "Outliers", "decision": str(OUTLIER_SIGMA) + " sigma bounds from the training set, treatment "
     + final_choice["outlier"]},
    {"step": "Scaling", "decision": "min-max, learned on the training set"},
    {"step": "Missing values", "decision": "kNN imputation, k = " + str(final_choice["k"])},
    {"step": "Sensors", "decision": str(len(selected_features)) + " Boruta sensors"},
    {"step": "Balancing", "decision": final_choice["balancing"] + " (outside the model)"},
    {"step": "Model", "decision": display_name(final_family) + ", " + final_choice["setting"] + " = "
     + str(final_choice["value"]) + ", seed " + str(SEED)},
    {"step": "Threshold", "decision": "fail if probability > " + str(final_threshold)},
    {"step": "Costs", "decision": "EUR " + str(COST_FALSE_POSITIVE) + " per false alarm, EUR "
     + str(COST_FALSE_NEGATIVE) + " per missed failure"},
    {"step": "Expected (cross-validated)", "decision": "cost EUR " + format(int(final_row["cost_eur"]), ",")
     + " for " + str(len(y_train)) + " wafers"}])
print(decision_sheet.to_string(index=False))
decision_sheet.to_csv(TABLES_FOLDER / "block17_final_decisions.csv", index=False)


# ----- 17.2 Apply the stored Phase 1 rules to the test set (no fitting) -----
print("\n--- 17.2 Test set: stored rules applied ---")
# The column list and the 3 sigma bounds were learned on the training set in
# Blocks 5 and 7; here they are only applied.
X_test_reduced = X_test.drop(columns=columns_removed_in_block5)
if list(X_test_reduced.columns) != list(X_train_reduced.columns):
    raise ValueError("The test set does not have the same columns as the training set.")
X_test_treated = apply_outlier_treatment(final_choice["outlier"], X_test_reduced, lower_bounds, upper_bounds)
print("Test set:", X_test.shape, "->", X_test_treated.shape, "| same columns as training: True")
print("Test values capped with the training bounds:", count_changed_values(X_test_reduced, X_test_treated),
      "| missing test values (filled by the imputer learned on training):", int(X_test_treated.isna().sum().sum()))
save_processed(X_test_treated, y_test, "block17_test_treated.csv")


# ----- 17.3 Refit the final pipeline on all training wafers -----
print("\n--- 17.3 Final pipeline fitted on all", len(y_train), "training wafers ---")
final_model = make_model(final_family, final_choice["balancing"])
final_model.set_params(**{final_choice["setting"]: final_choice["value"]})
final_pipeline = build_pipeline(final_model, final_choice["balancing"], n_neighbors=final_choice["k"])
# fit() is called on training data only: scaler, imputer and model learn here.
final_pipeline.fit(outlier_versions[final_choice["outlier"]], y_train)
print("Fitted:", display_name(final_family), "|", final_choice["setting"], "=", final_choice["value"],
      "| k =", final_choice["k"], "| threshold", final_threshold)


# ----- 17.4 Predict the test set (transform only), then open the labels -----
print("\n--- 17.4 FINAL TEST: predictions for the", len(y_test), "test wafers ---")
# final_pipeline[:-1] runs every step except the model; on new data the scaler,
# imputer and selector only transform (and the balancing step is skipped).
test_ready = final_pipeline[:-1].transform(X_test_treated)
print("Test data as the model sees it:", test_ready.shape, "(expected", len(y_test), "x",
      len(selected_features), ")")
test_probabilities = pd.Series(final_pipeline.predict_proba(X_test_treated)[:, 1], index=y_test.index)
test_predictions = (test_probabilities > final_threshold).astype(int)
# y_test is used here for the first time since Block 4.
test_result = full_evaluation(y_test, test_predictions, test_probabilities)
test_result["average_precision"] = round(float(average_precision_score(y_test, test_probabilities)), 4)
print("FINAL TEST:", test_result)
prediction_table = pd.DataFrame({"true_label": y_test, "failure_probability": test_probabilities.round(4),
                                 "predicted_label": test_predictions})
prediction_table.to_csv(TABLES_FOLDER / "block17_test_predictions.csv", index_label="wafer_id")


# ----- 17.5 Cross-validation (training) vs. final test -----
print("\n--- 17.5 Cross-validated training result vs. FINAL TEST ---")
cv_result = full_evaluation(y_train, final_predictions, final_probabilities)
cv_result["average_precision"] = round(float(average_precision_score(y_train, final_probabilities)), 4)
test_failures = int(y_test.sum())
test_passes = len(y_test) - test_failures
comparison_rows = []
for metric in ["TP", "FN", "FP", "TN", "accuracy", "recall", "specificity", "precision", "F1", "F2",
               "ROC_AUC", "average_precision", "cost_eur"]:
    comparison_rows.append({"metric": metric, "cross_validated_training": cv_result[metric],
                            "final_test": test_result[metric]})
# Cost per wafer makes the two sets (1,253 and 314 wafers) comparable.
comparison_rows.append({"metric": "cost_per_wafer_eur",
                        "cross_validated_training": round(cv_result["cost_eur"] / len(y_train), 1),
                        "final_test": round(test_result["cost_eur"] / len(y_test), 1)})
# Reference strategies on the test set (no model involved).
comparison_rows.append({"metric": "test: flag no wafer (EUR)", "cross_validated_training": "",
                        "final_test": test_failures * COST_FALSE_NEGATIVE})
comparison_rows.append({"metric": "test: flag every wafer (EUR)", "cross_validated_training": "",
                        "final_test": test_passes * COST_FALSE_POSITIVE})
final_comparison = pd.DataFrame(comparison_rows)
print(final_comparison.to_string(index=False))
final_comparison.to_csv(TABLES_FOLDER / "block17_final_test_results.csv", index=False)

# Figures: test confusion matrix; test ROC and precision-recall curves with the
# 0.49 point; cross-validation vs. test on the main measures.
plot_confusion_matrix(y_test, test_predictions, "FINAL TEST: " + display_name(final_family) + ", threshold "
                      + str(final_threshold) + " (" + str(len(y_test)) + " test wafers)",
                      "block17_01_confusion_matrix_test.png")
figure, axes = plt.subplots(1, 2, figsize=(13, 6))
false_positive_rate, true_positive_rate, roc_thresholds = roc_curve(y_test, test_probabilities)
axes[0].plot(false_positive_rate, true_positive_rate, color=FINAL_COLOR, linewidth=2.5,
             label="test (AUC " + format(test_result["ROC_AUC"], ".3f") + ")")
axes[0].plot([0, 1], [0, 1], linestyle="--", color=COLOR_REFERENCE, label="random guess")
axes[0].plot(1 - test_result["specificity"], test_result["recall"], marker="o", markersize=10, color=COLOR_FAIL,
             linestyle="none", label="threshold " + str(final_threshold))
axes[0].legend(fontsize=9, loc="lower right")
presentation_axis(axes[0], "ROC curve, FINAL TEST", "False-positive rate", "Recall (failed wafers caught)")
precision_values, recall_values, pr_thresholds = precision_recall_curve(y_test, test_probabilities)
axes[1].plot(recall_values, precision_values, color=FINAL_COLOR, linewidth=2.5,
             label="test (AP " + format(test_result["average_precision"], ".3f") + ")")
axes[1].axhline(test_failures / len(y_test), linestyle="--", color=COLOR_REFERENCE,
                label="random guess = failure share (" + format(test_failures / len(y_test), ".3f") + ")")
axes[1].plot(test_result["recall"], test_result["precision"], marker="o", markersize=10, color=COLOR_FAIL,
             linestyle="none", label="threshold " + str(final_threshold))
axes[1].legend(fontsize=9, loc="upper right")
presentation_axis(axes[1], "Precision-recall curve, FINAL TEST", "Recall", "Precision")
save_figure(figure, "block17_02_roc_pr_test.png")

measures = ["recall", "specificity", "precision", "ROC_AUC"]
figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [3, 1]})
positions = np.arange(len(measures))
cv_values = []
test_values = []
for measure in measures:
    cv_values.append(cv_result[measure])
    test_values.append(test_result[measure])
axes[0].bar(positions - 0.2, cv_values, width=0.4, color=OTHER_COLOR, label="cross-validated (training)")
axes[0].bar(positions + 0.2, test_values, width=0.4, color=FINAL_COLOR, label="FINAL TEST")
for position in range(len(measures)):
    axes[0].text(positions[position] - 0.2, cv_values[position] + 0.02, format(cv_values[position], ".2f"),
                 ha="center", fontsize=9)
    axes[0].text(positions[position] + 0.2, test_values[position] + 0.02, format(test_values[position], ".2f"),
                 ha="center", fontsize=9)
axes[0].set_xticks(positions)
axes[0].set_xticklabels(measures)
axes[0].set_ylim(0, 1.1)
axes[0].legend(fontsize=9)
presentation_axis(axes[0], "Main measures", "", "Value")
cost_per_wafer = [cv_result["cost_eur"] / len(y_train), test_result["cost_eur"] / len(y_test)]
axes[1].bar([0, 1], cost_per_wafer, color=[OTHER_COLOR, FINAL_COLOR])
for position in range(2):
    axes[1].text(position, cost_per_wafer[position] + 3, "EUR " + str(int(round(cost_per_wafer[position]))),
                 ha="center", fontsize=9)
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["CV", "test"])
presentation_axis(axes[1], "Cost per wafer", "", "EUR")
figure.suptitle(display_name(final_family) + " at threshold " + str(final_threshold)
                + ": cross-validated training result vs. FINAL TEST", fontsize=13)
save_figure(figure, "block17_03_cv_vs_test.png")


# ----- 17.6 Uncertainty: how precise is the test recall with 21 failures? -----
print("\n--- 17.6 95 % confidence range for recall (Wilson) ---")
test_low, test_high = wilson_interval(test_result["TP"], test_failures)
cv_low, cv_high = wilson_interval(cv_result["TP"], int(y_train.sum()))
print("FINAL TEST recall", test_result["recall"], "(" + str(test_result["TP"]) + " of " + str(test_failures)
      + "), 95 % range", round(test_low, 3), "to", round(test_high, 3))
print("Cross-validated recall", cv_result["recall"], "(" + str(cv_result["TP"]) + " of " + str(int(y_train.sum()))
      + "), 95 % range", round(cv_low, 3), "to", round(cv_high, 3))
pd.DataFrame([{"set": "cross-validated training", "caught": cv_result["TP"], "failed": int(y_train.sum()),
               "recall": cv_result["recall"], "low_95": round(cv_low, 3), "high_95": round(cv_high, 3)},
              {"set": "final test", "caught": test_result["TP"], "failed": test_failures,
               "recall": test_result["recall"], "low_95": round(test_low, 3), "high_95": round(test_high, 3)}]
             ).to_csv(TABLES_FOLDER / "block17_recall_confidence.csv", index=False)


# ----- 17.7 Lock -----
print("\n--- 17.7 The test set has been used. Nothing may be changed because of these numbers. ---")
print("Expected reasons for a gap (written before the run, DECISIONS.md): Boruta list chosen outside the folds,")
print("best of many settings chosen (winner's curse), only", test_failures, "failed wafers in the test set.")


# %% Block 18: Project summary, saved pipeline and reproducibility record
# Closes the project (manual Sections 15.5, 16 and 17.2): a before -> after
# summary of every stage and of the cost across the project, the final fitted
# pipeline saved once to a file, and a record of the data and software versions
# so that anyone can rerun the script and get the same numbers. Nothing is
# refitted or changed; the test result from Block 17 is only reported.


# ----- 18.1 What each stage changed (manual 16.1) -----
print("\n--- 18.1 What each stage changed (before -> after) ---")
# Outlier capping on the training set: changed values and wafers with at least one.
both_missing_train = X_train_reduced.isna() & X_train_treated.isna()
capped_mask = (X_train_treated != X_train_reduced) & ~both_missing_train
capped_values = int(capped_mask.sum().sum())
capped_wafers = int(capped_mask.any(axis=1).sum())
failed_training = int(y_train.sum())
stage_summary = pd.DataFrame([
    {"stage": "Data (Block 2)", "before": "raw UCI files",
     "after": str(X.shape[0]) + " wafers x " + str(X.shape[1]) + " sensors, " + str(int(y.sum())) + " failed"},
    {"stage": "Split (Block 4)", "before": str(len(X)) + " wafers",
     "after": str(len(y_train)) + " training (" + str(failed_training) + " failed) / " + str(len(y_test))
     + " test (" + str(int(y_test.sum())) + " failed)"},
    {"stage": "Column removal (Block 5)", "before": str(X_train.shape[1]) + " columns",
     "after": str(X_train_reduced.shape[1]) + " columns (" + str(len(removed_zero_variance)) + " zero variance, "
     + str(len(removed_duplicate_columns)) + " duplicates, " + str(len(removed_missing_columns))
     + " > " + str(MISSING_THRESHOLD) + " % missing)"},
    {"stage": "Outliers (Block 7)", "before": "3 sigma bounds from the training set",
     "after": str(capped_values) + " training values capped in " + str(capped_wafers) + " of " + str(len(y_train))
     + " wafers; " + str(count_changed_values(X_test_reduced, X_test_treated)) + " test values capped"},
    {"stage": "Missing values (Block 8)", "before": str(int(X_train_treated.isna().sum().sum())) + " training gaps",
     "after": str(int(X_train_imputed.isna().sum().sum())) + " after kNN imputation (k = " + str(KNN_NEIGHBORS)
     + "); mean imputation not used (rule 3)"},
    {"stage": "Scaling (Block 8)", "before": "sensors in their own units",
     "after": "min-max to 0..1, needed for kNN distances and logistic regression"},
    {"stage": "Feature selection (Block 10)", "before": str(X_train_reduced.shape[1]) + " sensors",
     "after": str(len(selected_features)) + " Boruta sensors (" + str(len(confirmed_features)) + " confirmed + "
     + str(len(tentative_features)) + " tentative)"},
    {"stage": "Balancing (final model)", "before": str(len(y_train) - failed_training) + " passed : "
     + str(failed_training) + " failed",
     "after": "inside EasyEnsemble: each of " + str(final_choice["value"]) + " sub-models learns from "
     + str(failed_training) + " failed + " + str(failed_training) + " randomly chosen passed wafers"},
    {"stage": "Model and threshold (Blocks 14-16)", "before": "22 combinations at threshold 0.5",
     "after": display_name(final_family) + ", threshold " + str(final_threshold)}])
print(stage_summary.to_string(index=False))
stage_summary.to_csv(TABLES_FOLDER / "block18_stage_summary.csv", index=False)


# ----- 18.2 Cost across the project, per wafer -----
print("\n--- 18.2 Cost across the project (cross-validated training, then the final test) ---")
# Each milestone's total cost, taken from the tables of the earlier blocks.
journey = [
    ("Random forest,\n450 sensors (B9)",
     float(selection_comparison[selection_comparison["feature_set"] == "all sensors (Block 9)"]["cost_eur"].iloc[0])),
    ("Random forest,\n18 sensors (B10)",
     float(selection_comparison[selection_comparison["feature_set"] == "Boruta selected (D10)"]["cost_eur"].iloc[0])),
    ("+ ROSE\n(B11)", float(balancing_comparison[balancing_comparison["option"] == "ROSE"]["cost_eur"].iloc[0])),
    ("Leak-proof\npipeline (B13)", float(pipeline_comparison["cost_eur"].iloc[1])),
    ("EasyEnsemble\nat 0.5 (B14)",
     float(model_matrix[model_matrix["combination"] == candidate_names[final_family]]["cost_at_0.5"].iloc[0])),
    ("Tuned, at 0.5\n(B15-16)", float(default_row["cost_eur"])),
    ("Tuned, at " + str(final_threshold) + "\n(B16)", float(final_row["cost_eur"]))]
journey_rows = []
for label, cost in journey:
    journey_rows.append({"stage": label.replace("\n", " "), "set": "cross-validated training",
                         "cost_eur": int(cost), "cost_per_wafer_eur": round(cost / len(y_train), 1)})
journey_rows.append({"stage": "FINAL TEST (B17)", "set": "test", "cost_eur": int(test_result["cost_eur"]),
                     "cost_per_wafer_eur": round(test_result["cost_eur"] / len(y_test), 1)})
journey_table = pd.DataFrame(journey_rows)
print(journey_table.to_string(index=False))
journey_table.to_csv(TABLES_FOLDER / "block18_cost_journey.csv", index=False)

# Figure: cost per wafer at each milestone; the final test in blue, and the
# cost of inspecting every wafer as a dashed reference line.
inspect_all_per_wafer = (len(y_train) - failed_training) * COST_FALSE_POSITIVE / len(y_train)
figure, axis = plt.subplots(figsize=(13, 6))
bar_labels = []
bar_values = []
bar_colors = []
for label, cost in journey:
    bar_labels.append(label)
    bar_values.append(cost / len(y_train))
    bar_colors.append(OTHER_COLOR)
bar_labels.append("FINAL TEST\n(B17)")
bar_values.append(test_result["cost_eur"] / len(y_test))
bar_colors.append(FINAL_COLOR)
axis.bar(range(len(bar_values)), bar_values, color=bar_colors)
for position in range(len(bar_values)):
    axis.text(position, bar_values[position] + 5, "EUR " + str(int(round(bar_values[position]))),
              ha="center", fontsize=10)
axis.axhline(inspect_all_per_wafer, linestyle="--", color=COLOR_REFERENCE)
axis.text(len(bar_values) - 0.5, inspect_all_per_wafer + 5, "inspect every wafer: EUR "
          + str(int(round(inspect_all_per_wafer))), ha="right", fontsize=9, color=COLOR_REFERENCE)
axis.set_xticks(range(len(bar_values)))
axis.set_xticklabels(bar_labels, fontsize=9)
presentation_axis(axis, "Cost per wafer across the project: from EUR " + str(int(round(bar_values[0])))
                  + " to EUR " + str(int(round(bar_values[-2]))) + " (cross-validated) and EUR "
                  + str(int(round(bar_values[-1]))) + " on the final test",
                  "", "Cost per wafer (EUR)")
save_figure(figure, "block18_01_cost_journey.png")


# ----- 18.3 Save the fitted final pipeline once (manual 17.2) -----
print("\n--- 18.3 Final pipeline saved ---")
# joblib.dump writes the fitted pipeline (scaler, imputer, sensor list, model)
# to one file; joblib.load gives it back without refitting. The file can only be
# loaded with the same package versions (requirements.txt).
MODELS_FOLDER = PROJECT_FOLDER / "results" / "models"
MODELS_FOLDER.mkdir(parents=True, exist_ok=True)
pipeline_file = MODELS_FOLDER / "block18_final_pipeline.joblib"
joblib.dump(final_pipeline, pipeline_file)
# Check: the reloaded pipeline gives exactly the same test probabilities.
reloaded_pipeline = joblib.load(pipeline_file)
reloaded_probabilities = reloaded_pipeline.predict_proba(X_test_treated)[:, 1]
print("Saved:", pipeline_file, "| size", round(pipeline_file.stat().st_size / 1024 / 1024, 2), "MB")
print("Reloaded pipeline gives identical test probabilities:",
      bool(np.array_equal(reloaded_probabilities, test_probabilities.to_numpy())))


# ----- 18.4 Reproducibility record (manual 17.2) -----
print("\n--- 18.4 Reproducibility record ---")
# SHA-256 turns each downloaded file into a 64-character fingerprint; the same
# file always gives the same fingerprint, and one changed byte changes it completely.
record_rows = [
    {"item": "run date (data accessed)", "value": str(datetime.date.today())},
    {"item": "produced on", "value": "local machine, " + platform.system() + " " + platform.release()},
    {"item": "Python", "value": platform.python_version()},
    {"item": "random seed (split, balancing, models, CV)", "value": str(SEED)},
    {"item": "costs (false alarm / missed failure, EUR)",
     "value": str(COST_FALSE_POSITIVE) + " / " + str(COST_FALSE_NEGATIVE)}]
for file_name, url, content in [("secom.data", URL_SENSOR_DATA, sensor_bytes),
                                ("secom_labels.data", URL_LABELS, label_bytes)]:
    record_rows.append({"item": "source " + file_name, "value": url})
    record_rows.append({"item": "size of " + file_name + " (bytes)", "value": str(len(content))})
    record_rows.append({"item": "SHA-256 of " + file_name, "value": hashlib.sha256(content).hexdigest()})
for package_name in REQUIRED_PACKAGES:
    record_rows.append({"item": "package " + package_name, "value": importlib.metadata.version(package_name)})
reproducibility_record = pd.DataFrame(record_rows)
print(reproducibility_record.to_string(index=False))
reproducibility_record.to_csv(TABLES_FOLDER / "block18_reproducibility.csv", index=False)


# ----- 18.5 Final summary -----
print("\n--- 18.5 Final summary ---")
print("Final model:", display_name(final_family), "(" + final_choice["setting"], "=", str(final_choice["value"]) + "),",
      "outliers", final_choice["outlier"] + ", k =", final_choice["k"], ",", len(selected_features), "sensors,",
      "threshold", final_threshold)
print("Cross-validated (training):", cv_result["TP"], "of", failed_training, "failures caught, cost EUR",
      format(cv_result["cost_eur"], ","), "(EUR", round(cv_result["cost_eur"] / len(y_train)), "per wafer)")
print("FINAL TEST:", test_result["TP"], "of", test_failures, "failures caught, recall", test_result["recall"],
      "| specificity", test_result["specificity"], "| ROC-AUC", test_result["ROC_AUC"], "| cost EUR",
      format(test_result["cost_eur"], ","), "(EUR", round(test_result["cost_eur"] / len(y_test)), "per wafer)")
print("\nPhase 3 finished.")

