import ast
import configparser
import glob
import math
import os
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd

from safelanding.config import *
from safelanding.labels import *
from safelanding.metrics import LzGtGenerator
from tqdm import tqdm


HEADLESS = True

basePath = Path(__file__).parents[2]
dataPath = basePath.joinpath("data", "imgs")
resultPath = basePath.joinpath("data", "results")
resultList = []

for dir in resultPath.iterdir():
    if dir.is_dir():
        resultList.append(dir)
resultList.sort()
resultList.pop(0)

for seq in tqdm(resultList):
    pathImgsGt = str(dataPath.joinpath(seq.stem, "gts", "*"))
    pathImgs = str(dataPath.joinpath(seq.stem, "images", "*"))

    config = configparser.ConfigParser()
    config.read(str(dataPath.joinpath(seq.stem, "config.ini")))
    dataset = config["SETTINGS"]["dataset"]
    labels = datasetLabels[dataset]
    r_landing = config["SETTINGS"]["r_landing"]

    gtsSeg = glob.glob(pathImgsGt)
    rgbImgs = glob.glob(pathImgs)
    gtsSeg.sort()
    rgbImgs.sort()
    df_lzs = pd.read_csv(
        str(resultPath.joinpath(seq.stem, "results_lzs.csv")),
        converters={"position": ast.literal_eval, 'id': lambda x: str(x)},
    )
    df_lzs['id'] = df_lzs['id'].astype(str)
    nbInferredImgs = df_lzs["id"].nunique()

    lzsGts = []

    if not (len(gtsSeg) == nbInferredImgs == len(rgbImgs)):
        raise IndexError("Non-matching number of ground truths/landing zones")

    for idx, gt in enumerate(gtsSeg):
        lzs = df_lzs[df_lzs["id"] == Path(gt).stem].to_dict("records")
        gtImg = cv.imread(gt)
        for lz in lzs:
            lz["gt"], lz["reasons"] = LzGtGenerator.getLzGt(gtImg, lz, labels)
        lzsGts += lzs

        if not HEADLESS:
            imgPath = rgbImgs[idx]
            img = cv.imread(imgPath)
            img = LzGtGenerator.draw_gt_lzs(img, lzs)
            cv.imshow("gt", img)
            cv.waitKey(0)
            cv.destroyAllWindows()

    dfLzsGt = pd.DataFrame(lzsGts)
    dfLzsGt.to_csv(resultPath.joinpath(seq.stem, "gt_lzs.csv"), index=False)
