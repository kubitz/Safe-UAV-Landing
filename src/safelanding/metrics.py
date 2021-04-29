import glob
import math
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd

from safelanding.labels import *

lzs = [
    {"confidence": 0.5, "position": [1000, 500], "radius": 100},
    {"confidence": 0.1, "position": [100, 500], "radius": 100},
]


def decodeRiskIds(crop, lb):
    """Checks if any class in the lz is considered unsafe

    :param crop: np array with all values equal to 0 except lz
    :type crop: np.array
    :param lb: bidirectional dictionary with labels and ids
    :type lb: bidict
    :return: if the zone is safe, True otherwise False
    :rtype: bool
    :return: if the zone is unsafe, list of reasons why, otherwise empty list
    :rtype: list
    """
    riskIds = np.unique(crop)
    isSafe = True
    reasons = []
    for riskId in riskIds:
        risk = lb.inverse[riskId][0]
        if risk in notSafe:
            isSafe = False
            reasons.append(risk)
    return isSafe, reasons


def getLzCrop(riskGt, lz):
    """Crops Gt to have just area of the landing Zone

    :param riskGt: 2d array containing gt labels
    :type riskGt: np.array.2D
    :param lz: landing zone data struct
    :type lz: lz
    :return: 2D np array with all values equal to zero except from lz
    :rtype: np.array.2D
    """
    posLz = lz.get("position")
    radiusLz = lz.get("radius")
    mask = np.zeros_like(riskGt)
    mask = cv.circle(mask, (posLz[0], posLz[1]), radiusLz, (255, 255, 255), -1)
    crop = cv.bitwise_and(riskGt, mask)
    return crop


def getLzGt(segImg, lz, lb):
    image, _, _ = cv.split(segImg)
    risk_gt = image.astype("uint8")
    crop = getLzCrop(risk_gt, lz)
    gt, reasons = decodeRiskIds(crop, lb)
    return gt, reasons


if __name__ == "__main__":

    dir_seg = (
        r"/home/kubitz/Documents/fyp/Safe-UAV-Landing/data/test/images/040003_017.jpg"
    )
    dir_seg = r"/home/kubitz/Documents/fyp/Safe-UAV-Landing/data/test/segmentation/040003_017.png"
    gtSeg = glob.glob("/content/Safe-UAV-Landing/data/test/seq1/images/*.jpg")
    lb = datasetLabels.get("aeroscapes")

    for i in range(len(gtSeg)):
        seg_img = cv.imread(dir_seg)
        fileName = Path(gtSeg[i]).stem
        lzs[i]["filename"] = fileName
        lzs[0]["gt"], lzs[0]["reasons"] = getLzGt(seg_img, lzs[i], lb)
    df = pd.DataFrame(lzs)
    df.to_csv("results.csv", index=False)
