import glob
from pathlib import Path
from typing import Sequence
import os
import time
import numpy as np
import pandas as pd
from cv2 import cv2 as cv
from PIL import Image

from safelanding.config import *
from safelanding.lzfinder import LzFinder
from safelanding.metrics import LzGtGenerator
SIMULATE=True
if not SIMULATE:
    from safelanding.seg_util import SegmentationEngine
    from safelanding.yolo_util import ObjectDetector

HEADLESS = True

if __name__ == "__main__":
    basePath = Path.cwd().parents[1]
    dataPath = str(basePath.joinpath("data", "imgs"))
    weigthObjPath = str(
        basePath.joinpath("data", "weights", "yolo-v3", "yolov3_leaky.weights")
    )
    cfgObjPath = str(
        basePath.joinpath("data", "weights", "yolo-v3", "yolov3_leaky.cfg")
    )
    namesObjPath = str(
        basePath.joinpath("data", "weights", "yolo-v3", "visdrone.names")
    )
    weigthSegPath = str(
        basePath.joinpath("data", "weights", "seg", "Unet-Mobilenet.pt")
    )
    segSimPath = str(basePath.joinpath("data", "imgs", "seq2", "masks", "*"))
    gtsPath = str(basePath.joinpath("data", "imgs", "seq2", "gts", "*"))
    imgsPath = str(basePath.joinpath("data", "imgs", "seq2", "images", "*"))
    resultPath = basePath.joinpath("data", "results")
    seq_name = Path(imgsPath).parts[-3]
    rgbImgs = glob.glob(imgsPath)
    dataFolders = []

    for dir in Path(dataPath).iterdir():
        if dir.is_dir():
            dataFolders.append(dir)
    dataFolders.sort()

    if SIMULATE:
        dataFolders = [dataFolders[0]]

    else:
        dataFolders.pop(0)
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
    for folder in dataFolders[:2]:
        seq_name = folder.stem
        print("starting with ", seq_name, " ...")
        segSimPath = str(basePath.joinpath("data", "imgs", folder.stem, "masks", "*"))
        gtsPath = str(basePath.joinpath("data", "imgs", folder.stem, "gts", "*"))
        imgsPath = str(basePath.joinpath("data", "imgs", folder.stem, "images", "*"))
        rgbImgs = glob.glob(imgsPath)
        segImgs = glob.glob(segSimPath)
        resultLzs = []
        detected_objs = []
        # Vectors to store execution times
        exec_pipeline = []
        exec_obj = []
        exec_seg = []
        exec_logic = []
        for i in range(len(rgbImgs)):
            start_pipeline = time.time()
            fileName = Path(rgbImgs[i]).stem
            img = cv.imread(rgbImgs[i])
            if not SIMULATE:
                height, width = img.shape[:2]
                img_pil = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_pil)
                start_seg = time.time()
                #segImg = segEngine.inferImage(img_pil)
                segImg = np.zeros((1152, 768,1), np.uint8)
                end_seg = time.time()
                segImg = np.array(segImg)
                start_obj = time.time()
                _, objs = objectDetector.infer_image(height, width, img)
                end_obj = time.time()
                obstacles = []
                for obstacle in objs:
                    print(obstacle)
                    posOb = obstacle.get("box")
                    minDist = 100
                    w, h = posOb[2], posOb[3]
                    obstacles.append(
                        [int(posOb[0] + w / 2), int(posOb[1] + h / 2), minDist]
                    )
                    obstacle["id"] = fileName
                    detected_objs.append(obstacle)
            else:
                segImg = cv.imread(segImgs[i])
                obstacles = []
            start_logic = time.time()
            lzFinder = LzFinder("graz", simulate=SIMULATE)
            lzs_ranked, risk_map = lzFinder.get_ranked_lz(
                obstacles, img, segImg, id=fileName, use_seg=False
            )
            end_logic = time.time()
            img = lzFinder.draw_lzs_obs(lzs_ranked[-5:], obstacles, img)
            resultLzs += lzs_ranked
            end_pipeline = time.time()

            exec_pipeline.append(end_pipeline-start_pipeline)
            exec_logic.append(end_logic-start_logic)
            if not SIMULATE:
                exec_obj.append(end_obj-start_obj)
                exec_seg.append(end_seg-start_seg)
            if EXTRACT_METRICS:
                gtSeg = glob.glob(gtsPath + "*.png")
                gtSeg.sort()
                if SIMULATE:
                    pass
            if SAVE_TO_FILE:
                Path.mkdir(
                    resultPath.joinpath(seq_name, "riskMaps"),
                    parents=True,
                    exist_ok=True,
                )
                Path.mkdir(
                    resultPath.joinpath(seq_name, "landingZones"),
                    parents=True,
                    exist_ok=True,
                )
                Path.mkdir(
                    resultPath.joinpath(seq_name, "masks"),
                    parents=True,
                    exist_ok=True,
                )
                cv.imwrite(
                    str(
                        resultPath.joinpath(
                            seq_name, "riskMaps", fileName + "_risk.jpg"
                        )
                    ),
                    cv.applyColorMap(risk_map, cv.COLORMAP_JET),
                )
                cv.imwrite(
                    str(
                        resultPath.joinpath(
                            seq_name, "landingZones", fileName + "_lzs.jpg"
                        )
                    ),
                    img,
                )
                cv.imwrite(
                    str(resultPath.joinpath(seq_name, "masks", fileName + "_mask.jpg")),
                    segImg,
                )

            if not HEADLESS:
                cv.imshow(
                    "best landing zones", cv.applyColorMap(risk_map, cv.COLORMAP_JET)
                )
                cv.waitKey(0)
                cv.imshow("best landing zones", img)
                cv.waitKey(0)
                cv.destroyAllWindows()

        if SAVE_TO_FILE:
            df_objDetect = pd.DataFrame(detected_objs)
            df_objDetect.to_csv(
                resultPath.joinpath(seq_name, "obj_detected.csv"), index=False
            )

        dfLzs = pd.DataFrame(resultLzs)
        dfLzs.to_csv(resultPath.joinpath(seq_name, "results_lzs.csv"), index=False)
        print("time pipeline: ", np.mean(exec_pipeline), " --- FPS:", 1/np.mean(exec_pipeline))
        print("time logic: ", np.mean(exec_logic), " --- FPS:", 1/np.mean(exec_logic))
        if not SIMULATE:
            print("time seg: ", np.mean(exec_seg), " --- FPS:", 1/np.mean(exec_seg))
            print("time obj: ", np.mean(exec_obj), " --- FPS:", 1/np.mean(exec_obj))
