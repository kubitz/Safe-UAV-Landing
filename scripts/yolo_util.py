import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
# code modified from https://github.com/iArunava/YOLOv3-Object-Detection-with-OpenCV


def show_image(img):
    cv2_imshow(img)
    cv.waitKey(0)

def draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels):
    # If there are any detections
    if len(idxs) > 0:
        for i in idxs.flatten():
            # Get the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            
            # Get the unique color for this class
            color = [int(c) for c in colors[classids[i]]]

            # Draw the bounding box rectangle and label on the image
            cv.rectangle(img, (x, y), (x+w, y+h), color, 2)
            text = "{}: {:4f}".format(labels[classids[i]], confidences[i])
            cv.putText(img, text, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img


def generate_boxes_confidences_classids(outs, height, width, tconf):
    boxes = []
    confidences = []
    classids = []

    for out in outs:
        for detection in out:
            #print (detection)
            #a = input('GO!')
            
            # Get the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]
            
            # Consider only the predictions that are above a certain confidence level
            if confidence > tconf:
                # TODO Check detection
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, bwidth, bheight = box.astype('int')

                # Using the center x, y coordinates to derive the top
                # and the left corner of the bounding box
                x = int(centerX - (bwidth / 2))
                y = int(centerY - (bheight / 2))

                # Append to list
                boxes.append([x, y, int(bwidth), int(bheight)])
                confidences.append(float(confidence))
                classids.append(classid)

    return boxes, confidences, classids




def infer_image(net, layer_names, height, width, img, colors, labels, 
            boxes=None, confidences=None, classids=None, idxs=None, infer=True, confidence=0.1, threshold=0.3, drawBoxes=True):
    
    if infer:
        # Contructing a blob from the input image
        blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), 
                        swapRB=True, crop=False)

        # Perform a forward pass of the YOLO object detector
        net.setInput(blob)

        # Getting the outputs from the output layers
        start = time.time()
        outs = net.forward(layer_names)
        end = time.time()

        print ("[INFO] YOLOv3 took {:6f} seconds".format(end - start))

        
        # Generate the boxes, confidences, and classIDs
        boxes, confidences, classids = generate_boxes_confidences_classids(outs, height, width, confidence)
        
        # Apply Non-Maxima Suppression to suppress overlapping bounding boxes
        idxs = cv.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

    if boxes is None or confidences is None or idxs is None or classids is None:
        raise '[ERROR] Required variables are set to None before drawing boxes on images.'
    
    obstacles=[]
    if len(idxs) > 0:
      for i in idxs.flatten():
          obstDetected = {
              'label': labels[classids[i]],
              'confidence':confidences[i],
              'box':boxes[i]
          }
          obstacles.append(obstDetected)     
    # Draw labels and boxes on the image
    if drawBoxes:
      img = draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels)
    return img, obstacles

pathLabels='/content/YOLO3-4-Py/tools/data/coco.names'
pathWeights="/content/09atO.weights"
pathCfg="/content/YOLO3-4-Py/tools/cfg/yolov3_leaky.cfg"
pathImg="/content/040005_089.jpg"
# Get the labels
labels = open(pathLabels).read().strip().split('\n')

# Intializing colors to represent each label uniquely
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# Load the weights and configutation to form the pretrained YOLOv3 model
net = cv.dnn.readNetFromDarknet(pathCfg, pathWeights)

# Get the output layer names of the model
layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

try:
  img = cv.imread(pathImg)
  height, width = img.shape[:2]
except:
  raise 'Image cannot be loaded!\n\
                            Please check the path provided!'

finally:
  img, obstacles = infer_image(net, layer_names, height, width, img, colors, labels)
  show_image(img)