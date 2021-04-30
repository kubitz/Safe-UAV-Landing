import glob
import math
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd

from safelanding.labels import *

lzs = [
    {"confidence": 0.5, "position": [600, 500], "radius": 100},
    {"confidence": 0.1, "position": [100, 500], "radius": 100},
]


class LzMetrics:
    def __init__(self):
        pass

    @classmethod
    def decodeRiskIds(cls, crop, lb):
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
        cv.imshow("image", cv.applyColorMap(crop, cv.COLORMAP_JET))
        cv.waitKey(0)
        cv.destroyAllWindows()
        isSafe = True
        reasons = []
        for riskId in riskIds:
            risk = lb.inverse[riskId][0]
            if risk in notSafe:
                isSafe = False
                reasons.append(risk)
        return isSafe, reasons

    @classmethod
    def getLzCrop(cls, riskGt, lz):
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

    @classmethod
    def getLzGt(cls, segImg, lz, lb):
        image, _, _ = cv.split(segImg)
        risk_gt = image.astype("uint8")
        crop = cls.getLzCrop(risk_gt, lz)
        gt, reasons = cls.decodeRiskIds(crop, lb)
        return gt, reasons


if __name__ == "__main__":
    dataset = "aeroscapes"
    dir_imgs = "/home/kubitz/Documents/fyp/Safe-UAV-Landing/data/imgs/seq1/images/*.jpg"
    dir_seg = "/home/kubitz/Documents/fyp/Safe-UAV-Landing/data/imgs/seq1/masks/*.jpg"
    dir_gtSeg = "/home/kubitz/Documents/fyp/Safe-UAV-Landing/data/imgs/seq1/gts/*.png"
    imgs = glob.glob(dir_imgs)
    segs = glob.glob(dir_seg)
    gtsSeg = glob.glob(dir_gtSeg)
    imgs.sort()
    segs.sort()
    gtsSeg.sort()

    lb = datasetLabels.get(dataset)

    for i in range(len(gtsSeg)):
        seg_img = cv.imread(gtsSeg[i])
        rgb = cv.imread(imgs[i])
        fileName = Path(gtsSeg[i]).stem
        lzs[i]["filename"] = fileName
        cv.imshow("image", LzMetrics.getLzCrop(rgb, lzs[i]))
        cv.waitKey(0)
        cv.destroyAllWindows()
        lzs[0]["gt"], lzs[0]["reasons"] = LzMetrics.getLzGt(seg_img, lzs[i], lb)

    df = pd.DataFrame(lzs)
    df.to_csv(
        "/home/kubitz/Documents/fyp/Safe-UAV-Landing/data/results.csv", index=False
    )
