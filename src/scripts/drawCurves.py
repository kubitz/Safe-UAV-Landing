import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame

from safelanding.metricsPlotter import BinaryClassification, Optimum

basePath = Path(__file__).parents[2]
dataPath = basePath.joinpath("data", "imgs")
resultPath = basePath.joinpath("data", "results")
resultList = []

for dir in resultPath.iterdir():
    if dir.is_dir():
        resultList.append(dir)

for seq in resultList:
    pathGt = str(resultPath.joinpath(seq.stem, "gt_lzs.csv"))
    basePath = Path(__file__).parents[2]
    df_lzs = pd.read_csv(
        pathGt,
        converters={"position": ast.literal_eval},
    )

    y_pred = df_lzs["confidence"].tolist()
    y_gt = df_lzs["gt"].tolist()
    opt = Optimum(np.array(y_gt), np.array(y_pred))
    report = opt.report()
    print(report)
    # Visualisation with plot_metric
    bc = BinaryClassification(y_gt, y_pred, labels=["Unsafe", "Safe"], threshold=0.61)
    bc.print_report()

    # Figures
    plt.figure(figsize=(15, 10))
    plt.subplot2grid(shape=(2, 6), loc=(0, 0), colspan=2)
    bc.plot_roc_curve()
    plt.subplot2grid((2, 6), (0, 2), colspan=2)
    bc.plot_precision_recall_curve()
    plt.subplot2grid((2, 6), (0, 4), colspan=2)
    bc.plot_class_distribution()
    plt.subplot2grid((2, 6), (1, 1), colspan=2)
    bc.plot_confusion_matrix()
    plt.subplot2grid((2, 6), (1, 3), colspan=2)
    bc.plot_confusion_matrix(normalize=True)

    # Save figure

    # Display Figure
    plt.show()
    plt.close()

    # Full report of the classification
