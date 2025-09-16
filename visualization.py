import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
import statistics as stats

# Load
file_path = "./experiments/baseline/test-results/results.csv"
folder_path = "./figures/data"
files = [
    # "3_robots_10_cr.csv",
    # "5_robots_10_cr.csv",
    # "7_robots_10_cr.csv",
    # "10_robots_10_cr.csv",
    # "12_robots_10_cr.csv",
    # "15_robots_10_cr.csv",
    # "18_robots_10_cr.csv",
    # "20_robots_5_cr.csv",
    # "20_robots_10_cr.csv",
    # "20_robots_10_cr_10_fov.csv",
    # "25_robots_10_cr.csv"
    "5fov.csv",
    "8fov.csv",
    "10fov.csv",
]


def sstd(xs):
    return stats.stdev(xs) if len(xs) > 1 else float("nan")

results = {}

for file in files:
    df = pd.read_csv(os.path.join(folder_path, file))
    makespan = df["Makespan"].values
    coverage = df["Coverage"].values
    Inference_Time_ms = df["Inference_Time_ms"].values
    Communication_Ratio = df["Communication_Ratio"].values

    avg_makespan = round(sum(makespan) / len(makespan), 2)
    avg_coverage = round(sum(coverage) / len(coverage), 2)
    avg_time = round(sum(Inference_Time_ms) / len(Inference_Time_ms), 2)
    avg_communication = round(sum(Communication_Ratio) / len(Communication_Ratio), 2)

    std_makespan = sstd(makespan)
    std_coverage = sstd(coverage)
    std_time = sstd(Inference_Time_ms)
    std_communication = sstd(Communication_Ratio)

    results[file] = {
        "makespan": (avg_makespan, std_makespan),
        "coverage": (avg_coverage, std_coverage),
        "time": (avg_time, std_time),
        "communication": (avg_communication, std_communication),
    }


our_files = [
    # "5_robots_10_cr.csv",
    # "7_robots_10_cr.csv",
    # "10_robots_10_cr.csv",
    # "12_robots_10_cr.csv",
    # "15_robots_10_cr.csv",
    # "18_robots_10_cr.csv",
    # "20_robots_5_cr.csv",
    # "20_robots_10_cr.csv",
    # "20_robots_10_cr_10_fov.csv",
    # "25_robots_10_cr.csv"
]

for file in our_files:
    df = pd.read_csv(os.path.join(folder_path, file))
    avg_makespan, std_makespan = df["Avg_Duration_Steps"].values, df["Std_Duration_Steps"].values
    avg_coverage, std_coverage = df["Avg_Coverage_Percent"].values, df["Std_Coverage_Percent"].values
    avg_time, std_time = df["Avg_Time"].values * 1000.0, df["Std_Time"].values * 1000.0
    avg_communication, std_communication = df["Avg_Connection_Percent"].values, 0.0


    results[file] = {
        "makespan": (avg_makespan[0], std_makespan[0]),
        "coverage": (avg_coverage[0], std_coverage[0]),
        "time": (avg_time[0], std_time[0]),
        "communication": (avg_communication[0], std_communication),
    }
df = pd.read_csv(file_path).sort_values("Num_Robots").reset_index(drop=True)

# Categorical x positions (0..4) and custom labels
x = np.arange(len(files) + len(our_files))                # 0..N-1
x_labels = []

makespans = [[], []]
coverages = [[], []]
times = [[], []]
communications = [[], []]
for f in files:
    x_labels.append(f.split("_")[0])
    makespans[0].append(results[f]["makespan"][0])
    makespans[1].append(results[f]["makespan"][1])
    coverages[0].append(results[f]["coverage"][0])
    coverages[1].append(results[f]["coverage"][1])
    times[0].append(results[f]["time"][0])
    times[1].append(results[f]["time"][1])
    communications[0].append(results[f]["communication"][0])
    communications[1].append(results[f]["communication"][1])
for f in our_files:
    x_labels.append(f.split("_")[0])
    makespans[0].append(results[f]["makespan"][0])
    makespans[1].append(results[f]["makespan"][1])
    coverages[0].append(results[f]["coverage"][0])
    coverages[1].append(results[f]["coverage"][1])
    times[0].append(results[f]["time"][0])
    times[1].append(results[f]["time"][1])
    communications[0].append(results[f]["communication"][0])
    communications[1].append(results[f]["communication"][1])
# labels = x_labels
labels = ["3", "8", "10"]

metrics = [
    "Coverage",
    "Communication Ratio",
    "Makespan",
    "Time",
]

for title in metrics:
    plt.figure(figsize=(8, 5))

    if title == "Communication Ratio":
        plt.bar(x, communications[0], capsize=12, width=0.8)
    elif title == "Coverage":
        plt.bar(x, coverages[0], yerr=coverages[1], capsize=12, width=0.8)
    elif title == "Time":
        plt.bar(x, times[0], yerr=times[1], capsize=12, width=0.8)
    elif title == "Makespan":
        plt.bar(x, makespans[0], yerr=makespans[1], capsize=12, width=0.8)

    # plt.xlabel("Configuration", fontsize=18)
    plt.xlabel("Sensing Range", fontsize=18)
    plt.ylabel(title, fontsize=21)
    plt.title(f"", fontsize=21)
    plt.xticks(x, labels, rotation=0, fontsize=18)         
    plt.yticks(fontsize=18)
    plt.xlim(-0.5, len(labels) - 0.5)

    plt.margins(x=0)                          
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"./figures/{title}.png", transparent=True)
