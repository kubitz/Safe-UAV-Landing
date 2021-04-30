import glob
from pathlib import Path

import numpy as np
from cv2 import cv2 as cv
from PIL import Image

from safelanding.config import *
from safelanding.lzfinder import LzFinder
from safelanding.metrics import LzMetrics

if not SIMULATE:
    from safelanding.seg_util import SegmentationEngine
    from safelanding.yolo_util import ObjectDetector

if __name__ == "__main__":

    basePath = Path(__file__).parents[2]
    dataPath = str(basePath.joinpath("data", "imgs"))
    weigthObjPath = str(
        basePath.joinpath("data", "weights", "yolo-v3", "yolov3_leaky.weights")
    )
    cfgObjPath = str(basePath.joinpath("data", "weights", "yolov3_leaky.cfg"))
    namesObjPath = str(
        basePath.joinpath("data", "weights", "yolo-v3", "visdrone.names")
    )
    weigthSegPath = str(
        basePath.joinpath("data", "weights", "seg", "Unet-Mobilenet.pt")
    )
    segSimPath = str(basePath.joinpath("data", "imgs", "0_simulation", "masks", "*"))
    gtsPath = str(basePath.joinpath("data", "imgs", "0_simulation", "gts", "*"))
    imgsPath = str(basePath.joinpath("data", "imgs", "0_simulation", "images", "*"))
    resultPath = basePath.joinpath("data","results")

    rgbImgs = glob.glob(imgsPath)
    if not SIMULATE:
        objectDetector = ObjectDetector(namesObjPath, weigthObjPath, cfgObjPath)
        segEngine = SegmentationEngine(weigthSegPath)
    else:
        segImgs = glob.glob(segSimPath)
        segImgs.sort()
        rgbImgs.sort()
        seq_obstacles = [
            [(640, 330, 100)],
            [(643, 346, 100)],
            [(638, 365, 100)],
            [(643, 387, 100)],
            [(645, 398, 100)],
            [(642, 414, 100)],
            [(640, 437, 100)],
            [(638, 468, 100)],
            [(643, 488, 100)],
            [(640, 488, 100)],
        ]

    for i in range(len(rgbImgs)):
        fileName = Path(rgbImgs[i]).stem
        img = cv.imread(rgbImgs[i])
        if not SIMULATE:
            height, width = img.shape[:2]
            img_pil = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_pil)
            segImg = segEngine.inferImage(img_pil)
            segImg = np.array(segImg)
            _, objs = objectDetector.infer_image(height, width, img)
            obstacles = []
            for obstacle in objs:
                print(obstacle)
                posOb = obstacle.get("box")
                minDist = 100
                w, h = posOb[2], posOb[3]
                obstacles.append(
                    [int(posOb[0] + w / 2), int(posOb[1] + h / 2), minDist]
                )
        else:
            segImg = cv.imread(segImgs[i])
            obstacles = seq_obstacles[i]

        lzFinder = LzFinder("aeroscapes")
        lzs_ranked, risk_map = lzFinder.get_ranked_lz(obstacles, img, segImg)
        img = lzFinder.draw_lzs_obs(lzs_ranked[-5:], obstacles, img)

        if EXTRACT_METRICS:
            gtSeg = glob.glob(gtsPath + "*.png")
            gtSeg.sort()
            if SIMULATE:
                print("implement metrics")
        if SAVE_TO_FILE:
            cv.imwrite(
                str(resultPath.joinpath("riskMaps", fileName + "_risk.jpg")),
                cv.applyColorMap(risk_map, cv.COLORMAP_JET),
            )
            cv.imwrite(
                str(resultPath.joinpath("landingZones", fileName + "_lzs.jpg")),
                img,
            )

        if not HEADLESS:
            cv.imshow("best landing zones", cv.applyColorMap(risk_map, cv.COLORMAP_JET))
            cv.waitKey(0)
            cv.imshow("best landing zones", img)
            cv.waitKey(0)
            cv.destroyAllWindows()
