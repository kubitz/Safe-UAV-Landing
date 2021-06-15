import glob
import math
from pathlib import Path
import json 
import cv2 as cv
import numpy as np
import pandas as pd

basePath = Path("/home/kubitz/Documents/fyp/results/baseline/results")
dataPath = Path("/home/kubitz/Documents/fyp/Safe-UAV-Landing/data/imgs")
folders = []
for dir in Path(dataPath).iterdir():
    if dir.is_dir():
        folders.append(dir)
folders.sort()
folders.pop(0)

print(folders)

for seq in folders:
    pathImgsGt = str(dataPath.joinpath(seq.stem, "gts", "*"))
    gtsSeg = glob.glob(pathImgsGt)
    num_class = -1
    for imgPath in gtsSeg:
        print(Path(imgPath).stem)
        img=cv.imread(imgPath)
        n = len(np.unique(img))
        num_class = max(num_class,n)
    print("seq: ", num_class)

