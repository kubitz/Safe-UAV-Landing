import glob
import math
from pathlib import Path
import os
import cv2 as cv
import numpy as np
import pandas as pd
import configparser

from safelanding.labels import *
from safelanding.metrics import LzMetrics
from safelanding.config import *

basePath = Path(__file__).parents[2]
dataPath = basePath.joinpath("data", "imgs")
dirList = []

for dir in dataPath.iterdir():
    dirList.append(dir)

if SIMULATE:
    dirList.sort()
    dirList.pop(0)

for seq in dirList:
    pathImgs = str(seq.joinpath("gts", "*"))
    config = configparser.ConfigParser()
    config.read(seq.joinpath("config.ini"))
    dataset = config["SETTINGS"]["dataset"]
    r_landing = config["SETTINGS"]["r_landing"]
    gtsSeg = glob.glob(pathImgs)
