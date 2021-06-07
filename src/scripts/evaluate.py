import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame

from safelanding.metricsPlotter import BinaryClassification, Metrics, MetricsLz, Optimum

basePath = Path("/home/kubitz/Documents/fyp/results")
folders = []
for dir in Path(basePath).iterdir():
    if dir.is_dir():
        folders.append(dir)
folders.sort()
folders.pop(1)

for folder in folders:
    print("Processing: ", folder.stem)
    resultPath = folder.joinpath("results")
    resultList = []

    for dir in resultPath.iterdir():
        if dir.is_dir():
            resultList.append(dir)

    resultList.sort()
    resultList.pop(0)
    df_lzs = DataFrame()

    for seq in resultList:
        pathGt = str(resultPath.joinpath(seq.stem, "gt_lzs.csv"))
        basePath = Path(__file__).parents[2]
        df_lz = pd.read_csv(
            pathGt,
            converters={"position": ast.literal_eval, "reasons": ast.literal_eval},
        )
        df_lzs = pd.concat([df_lzs, df_lz], ignore_index=True)

    threshold = 0.8
    opt = MetricsLz(df_lzs, threshold=threshold)

    print("reasons: ", opt.reasonsFP)

    # Visualisation with plot_metric
    y_pred = df_lzs["confidence"].tolist()
    y_gt = df_lzs["gt"].tolist()
    bc = BinaryClassification(y_gt, y_pred, labels=["Unsafe", "Safe"], threshold=threshold)
    bc.print_report()

    # Figures
    plt.figure(figsize=(15, 10))
    plt.subplot2grid(shape=(2, 6), loc=(0, 0), colspan=2)
    bc.plot_roc_curve()
    plt.subplot2grid((2, 6), (0, 2), colspan=2)
    bc.plot_precision_recall_curve()
    plt.subplot2grid((2, 6), (0, 4), colspan=2)
    bc.plot_threshold()
    plt.subplot2grid((2, 6), (1, 1), colspan=2)
    bc.plot_confusion_matrix()
    plt.subplot2grid((2, 6), (1, 3), colspan=2)
    bc.plot_confusion_matrix(normalize=True)

    # Save figure
    plt.savefig(str(folder.parents[1].joinpath("results","gts","figures",folder.stem+"_fig.png")))
    # Display Figure
    plt.show()
    plt.close()

    # Full report of the classification
