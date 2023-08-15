import csv
import os
import cv2
import torch
import pandas as pd


def get_output_format(detections):
    # define output list
    output = []
    # define desired classes
    target_classes = [2, 3, 5, 7]  # is currently set to car, motorbike, bus and truck
    # unpack the tuple to get individual arrays
    class_ids, scores, boxes = detections
    for (classId, score, box) in zip(class_ids, scores, boxes):
        # only pass bounding boxes if they are a target class
        if classId in target_classes:
            if classId == 2:
                classId = 'car'
            elif classId == 3:
                classId = 'motorbike'
            elif classId == 5:
                classId = 'bus'
            elif classId == 7:
                classId = 'truck'
            output.append((box, score, classId))
    return output


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def xywh2xyxy(xywh):
    x, y, w, h = xywh
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    return x1, y1, x2, y2


images_path = 'inputs/detector_test/images'
images = []
for filename in os.listdir(images_path):
    img = cv2.imread(os.path.join(images_path, filename))
    images.append(img)

labels_path = 'inputs/detector_test/labels/'
labels_dfs = []
for filename in os.listdir(labels_path):
    df = pd.read_csv(os.path.join(labels_path, filename))
    labels_dfs.append(df)

# yolov4
# read in the network from the saved config and weight files
net = cv2.dnn.readNetFromDarknet('yolo_config_files/yolov4.cfg', 'yolo_config_files/yolov4.weights')
# set the network to be a detection model
yolov4 = cv2.dnn_DetectionModel(net)
# set the image size and input params
yolov4.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
# do detection on each image
yolov4_detections = []
for img in images:
    detections = yolov4.detect(img, nmsThreshold=0.4)
    # format the detections properly
    detections = get_output_format(detections)
    yolov4_detections.append(detections)

# calculate iou
# iterate over each image gt labels
ious = []
for labels, detections in zip(labels_dfs, yolov4_detections):
    # labels is all the label for a specific image
    # detections is all the detections for a specific image
    idx = 0
    for detection in detections:
        detection_bbx = xywh2xyxy(detection[0])
        best_iou = 0
        best_iou_class_match = 0
        for index, label in labels.iterrows():
            label_bbx = [label['xmin'], label['ymin'], label['xmax'], label['ymax']]
            # do iou
            iou = bb_intersection_over_union(label_bbx, detection_bbx)
            # if its the best iou then save the value and class
            if iou > best_iou:
                best_iou = iou
                if detection[2] == label['class']:
                    best_iou_class_match = 1

        detections[idx] = list(detections[idx])
        detections[idx].append(best_iou)
        detections[idx].append(best_iou_class_match)

        idx = idx + 1

total_detections = 0
true_class_count = 0
total_iou = 0
for detections in yolov4_detections:
    total_detections = total_detections + len(detections)
    for detection in detections:
        true_class_count = true_class_count + detection[4]
        total_iou = total_iou + detection[3]

class_accuracy = true_class_count / total_detections
iou_accuracy = total_iou / total_detections
print(class_accuracy)
print(iou_accuracy)
