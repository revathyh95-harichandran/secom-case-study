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
# (more of scikit-learn, imbalanced-learn, Boruta) when first needed.
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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

print("Figures are saved to:", FIGURES_FOLDER)
print("Tables are saved to: ", TABLES_FOLDER)

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
# Nothing is removed or changed in the sensor values here.

# Read the sensor readings. Values are separated by a single space (sep=" ";
# the pandas default would be a comma) and there is no header row (header=None;
# by default pandas would use the first wafer as column names).
# na_values=["NaN"] states explicitly that the text "NaN" marks a missing
# reading, so every gap becomes a real missing value (rule 5) instead of
# relying silently on pandas' default list of missing-value markers.
sensor_data = pd.read_csv(URL_SENSOR_DATA, sep=" ", header=None, na_values=["NaN"])

# Read the labels file: column 1 is the label, column 2 the test date and time.
# names= gives the two columns their names directly. The date-time contains a
# space but stays one field, because it is written in double quotes and '"' is
# pandas' default quote character.
labels = pd.read_csv(URL_LABELS, sep=" ", header=None, names=["target", "timestamp"])

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
# Variances range over many orders of magnitude, so the histogram uses log10:
# -2 means 0.01, 0 means 1, 2 means 100. The log of 0 does not exist, so the
# zero-variance columns are left out of this histogram and named in the title.
nonzero_variances = column_variances[column_variances > 0]
log_variances = np.log10(nonzero_variances)
figure, axis = plt.subplots(figsize=(8, 4))
axis.hist(log_variances, bins=40, color=COLOR_MAIN, edgecolor="white")
axis.set_xlabel("Column variance, log10 scale (-2 = 0.01, 0 = 1, 2 = 100)")
axis.set_ylabel("Number of columns")
axis.set_title("Variance per column, full data (" + str(len(zero_variance_columns))
               + " zero-variance columns not shown)")
axis.spines["top"].set_visible(False)
axis.spines["right"].set_visible(False)
save_figure(figure, "block3_02_variance_histogram.png")

# Pareto chart: for each candidate threshold, how many columns have a variance
# at or below it. np.logspace(-8, 8, 33) gives 33 thresholds from 10^-8 to 10^8,
# two per power of ten, matching the range of the variances.
variance_thresholds = np.logspace(-8, 8, 33)
columns_at_or_below = []
for threshold in variance_thresholds:
    columns_at_or_below.append((column_variances <= threshold).sum())
variance_pareto = pd.DataFrame({"variance_threshold": variance_thresholds,
                                "columns_at_or_below": columns_at_or_below})
variance_pareto.to_csv(TABLES_FOLDER / "block3_variance_pareto.csv", index=False)

figure, axis = plt.subplots(figsize=(8, 4))
axis.plot(variance_thresholds, columns_at_or_below, marker="o", markersize=4,
          linewidth=2, color=COLOR_MAIN)
axis.set_xscale("log")
axis.axhline(len(zero_variance_columns), linestyle="--", color=COLOR_REFERENCE)
axis.text(variance_thresholds[-1], len(zero_variance_columns) + 10,
          str(len(zero_variance_columns)) + " zero-variance columns ",
          ha="right", va="bottom", color=COLOR_REFERENCE)
axis.set_xlabel("Variance threshold (log scale)")
axis.set_ylabel("Columns with variance at or below threshold")
axis.set_title("Low-variance Pareto chart, full data (analysis only, no action)")
axis.spines["top"].set_visible(False)
axis.spines["right"].set_visible(False)
save_figure(figure, "block3_03_variance_pareto.png")

# Print the count at each power of ten (every second threshold).
for position in range(0, len(variance_thresholds), 2):
    print("  variance <=", variance_thresholds[position], ":", columns_at_or_below[position], "columns")


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

# (a) The manual's "histogram": one bar per column (x = column, y = % missing),
# sorted from most to least missing so the pattern is visible at a glance.
figure, axis = plt.subplots(figsize=(10, 4))
column_positions = range(len(missing_percent_sorted))
axis.bar(column_positions, missing_percent_sorted.values, width=1.0, color=COLOR_MAIN)
for reference_percent in [55, 60]:
    axis.axhline(reference_percent, linestyle="--", linewidth=1, color=COLOR_REFERENCE)
axis.text(len(missing_percent_sorted), 55, "55 % (papers) ", ha="right", va="top", color=COLOR_REFERENCE)
axis.text(len(missing_percent_sorted), 60, "60 % (manual) ", ha="right", va="bottom", color=COLOR_REFERENCE)
axis.set_xlabel("Sensor columns, sorted from most to least missing (" + str(len(X.columns)) + " columns)")
axis.set_ylabel("% missing")
axis.set_title("Missing values per column, full data")
axis.spines["top"].set_visible(False)
axis.spines["right"].set_visible(False)
save_figure(figure, "block3_04_missing_per_column.png")

# (b) Pareto chart: for each threshold 0, 5, ..., 100 %, how many columns have
# MORE missing values than the threshold, i.e. would be removed at that threshold.
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
axis.set_title("Missingness Pareto chart, full data (no threshold chosen here)")
axis.spines["top"].set_visible(False)
axis.spines["right"].set_visible(False)
save_figure(figure, "block3_05_missing_pareto.png")

for position in range(len(missing_thresholds)):
    print("  missing >", missing_thresholds[position], "% :", columns_removed[position], "columns")


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

# Count the pairs with |r| >= 0.8 (the papers' removal rule, reported only).
# The loop visits each pair once (j starts after i); missing r values count as no.
absolute_correlations = correlation_matrix.abs().values
number_of_columns = len(non_constant_columns)
strong_pairs = 0
for i in range(number_of_columns):
    for j in range(i + 1, number_of_columns):
        if absolute_correlations[i, j] >= 0.8:
            strong_pairs = strong_pairs + 1
print("Non-constant columns in the heatmap:", number_of_columns)
print("Pairs with |r| >= 0.8:", strong_pairs)

# Diverging colours: blue = negative, white = 0, red = positive, fixed from -1
# to +1 so colours mean the same in every heatmap. Pairs without enough data
# (no r) are shown in light grey via set_bad().
heatmap_colors = plt.get_cmap("RdBu_r").copy()
heatmap_colors.set_bad("lightgrey")
figure, axis = plt.subplots(figsize=(9, 8))
heatmap_image = axis.imshow(correlation_matrix.values, cmap=heatmap_colors, vmin=-1, vmax=1)
colour_bar = figure.colorbar(heatmap_image, ax=axis, shrink=0.8)
colour_bar.set_label("Pearson correlation r")
axis.set_xticks([])
axis.set_yticks([])
axis.set_xlabel("Sensor columns (" + str(number_of_columns) + " non-constant, original order)")
axis.set_ylabel("Sensor columns")
axis.set_title("Correlation heatmap, full data (" + str(strong_pairs) + " pairs with |r| >= 0.8)")
save_figure(figure, "block3_08_correlation_heatmap.png")

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
