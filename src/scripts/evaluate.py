import glob
import math
import ast
from pathlib import Path
import os
import cv2 as cv
import numpy as np
import pandas as pd
import configparser

from safelanding.labels import *
from safelanding.metrics import LzMetrics
from safelanding.config import *

HEADLESS=False

basePath = Path(__file__).parents[2]
dataPath = basePath.joinpath("data", "imgs")
resultPath = basePath.joinpath("data","results")
resultList = []

for dir in resultPath.iterdir():
    if dir.is_dir():
        resultList.append(dir)

for seq in resultList:
    pathImgsGt = str(dataPath.joinpath(seq.stem,"gts", "*"))
    pathImgs = str(dataPath.joinpath(seq.stem,"images", "*"))


    config = configparser.ConfigParser()
    config.read(str(dataPath.joinpath(seq.stem,"config.ini")))
    dataset = config["SETTINGS"]["dataset"]
    labels = datasetLabels[dataset]
    r_landing = config["SETTINGS"]["r_landing"]
    
    gtsSeg = glob.glob(pathImgsGt)
    rgbImgs = glob.glob(pathImgs)
    gtsSeg.sort()
    rgbImgs.sort()
    df_lzs=pd.read_csv(str(resultPath.joinpath(seq.stem,"results_lzs.csv")), converters={"position": ast.literal_eval})
    nbInferredImgs=df_lzs['id'].nunique()
    
    lzsGts=[]
    
    if not (len(gtsSeg)==nbInferredImgs==len(rgbImgs)):
        raise IndexError("Non-matching number of gts/lzs")

    for idx, gt in enumerate(gtsSeg):
        lzs=df_lzs[df_lzs["id"]==Path(gt).stem].to_dict("records")
        gtImg = cv.imread(gt)
        for lz in lzs:
            lz["gt"], lz["reasons"] = LzMetrics.getLzGt(gtImg, lz, labels)
        lzsGts.append(lzs)
    
        if not HEADLESS:
            imgPath=rgbImgs[idx]
            img=cv.imread(imgPath)
            img=LzMetrics.draw_gt_lzs(img,lzs)
            cv.imshow("gt",img)
            cv.waitKey(0)
            cv.destroyAllWindows()