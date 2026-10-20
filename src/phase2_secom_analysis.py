# %% Block 0: Header, package installation and imports
# SECOM wafer fault prediction - PHASE 2 SCRIPT (Part B, Presentation 2)
# Blocks 0 to 7 (Phase 1, unchanged): installation, settings, data download,
# data understanding, stratified split, feature reduction, correlation, outliers.
# Blocks 8 to 12 (Phase 2): missing-value check, scaling and kNN imputation,
# first random forest, Boruta feature selection, balancing, evaluation. Each phase has
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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Phase 2 imports: scaling and kNN imputation (Block 8).
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
# Phase 2 imports: first model and cross-validation (Block 9).
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix


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


# ----- 9.2 5-fold stratified cross-validation (written as a visible loop) -----
print("\n--- 9.2 Random forest, " + str(CV_FOLDS) + "-fold stratified cross-validation ---")
# StratifiedKFold cuts the training wafers into CV_FOLDS parts with the same
# failure share. shuffle=True mixes the wafers first (they are in time order);
# random_state=SEED makes the parts identical on every run.
cv_splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

# Each wafer gets exactly one out-of-fold prediction: from the model trained
# without its part.
cv_predictions = pd.Series(0, index=y_train.index)
cv_probabilities = pd.Series(0.0, index=y_train.index)
fold_rows = []
fold_number = 0
for train_positions, validation_positions in cv_splitter.split(X_train_imputed, y_train):
    fold_number = fold_number + 1
    # iloc selects rows by position: 4 parts to learn from, 1 part to evaluate on.
    X_fold_train = X_train_imputed.iloc[train_positions]
    y_fold_train = y_train.iloc[train_positions]
    X_fold_validation = X_train_imputed.iloc[validation_positions]
    y_fold_validation = y_train.iloc[validation_positions]

    # A new forest with the same settings for every fold, trained without the validation part.
    fold_model = RandomForestClassifier(random_state=SEED)
    fold_model.fit(X_fold_train, y_fold_train)
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

# %% Block 10: Boruta feature selection
# Select the informative sensors with Boruta on the training set (Phase 2).

# %% Block 11: Balancing
# Balance the training set with SMOTE or ROSE; the test set is never balanced (Phase 2).

# %% Block 12: Evaluation helpers
# Functions for the confusion matrix, metrics, ROC curve and total cost (Phase 2).
