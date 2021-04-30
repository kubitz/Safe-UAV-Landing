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
    PATH_GT = "/home/kubitz/Documents/fyp/Safe-UAV-Landing/data/imgs/seq1/gts"
    PATH_IMG_COLAB = "/content/Safe-UAV-Landing/data/imgs/seq1/images/*.jpg"
    PATH_IMG = "/home/kubitz/Documents/fyp/Safe-AV-Landing/data/imgs/seq1/images/*.jpg"
    PATH_SIM_DATA = (
        "/home/kubitz/Documents/fyp/Safe-UAV-Landing/data/imgs/seq1/masks/*.jpg"
    )
    PATH_SIM_DATA_COLAB = (
        "/home/kubitz/Documents/fyp/Safe-UAV-Landing/data/imgs/seq1/masks/*.jpg"
    )
    PATH_OBJ_DETECT_WEIGHTS = (
        "/content/Safe-UAV-Landing/data/weights/yolo-v3/yolov3_leaky.weights"
    )
    PATH_OBJ_CFG = "/content/Safe-UAV-Landing/data/weights/yolo-v3/yolov3_leaky.cfg"
    PATH_OBJ_NAMES = "/content/Safe-UAV-Landing/data/weights/yolo-v3/visdrone.names"
    PATH_SEG_WEIGHTS = "/content/Safe-UAV-Landing/data/weights/seg/Unet-Mobilenet.pt"
    rgbImgs = glob.glob(PATH_IMG_COLAB)
    rgbImgs = glob.glob(PATH_IMG)
    if not SIMULATE:
        objectDetector = ObjectDetector(
            PATH_OBJ_NAMES, PATH_OBJ_DETECT_WEIGHTS, PATH_OBJ_CFG
        )
        segEngine = SegmentationEngine(PATH_SEG_WEIGHTS)
    else:
        segImgs = glob.glob(PATH_SIM_DATA)
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
            gtSeg = glob.glob(PATH_GT + "*.png")
            if SIMULATE:
                print("Metrics to be implemented")

        if SAVE_TO_FILE:
            cv.imwrite(
                "/content/Safe-UAV-Landing/data/results/riskMaps/"
                + fileName
                + "_risk.jpg",
                cv.applyColorMap(risk_map, cv.COLORMAP_JET),
            )
            cv.imwrite(
                "/content/Safe-UAV-Landing/data/results/landingZones/"
                + fileName
                + "_lzs.jpg",
                img,
            )

        if not HEADLESS:
            cv.imshow("best landing zones", cv.applyColorMap(risk_map, cv.COLORMAP_JET))
            cv.waitKey(0)
            cv.imshow("best landing zones", img)
            cv.waitKey(0)
            cv.destroyAllWindows()
