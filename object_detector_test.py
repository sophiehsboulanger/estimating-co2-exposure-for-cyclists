import csv
import os
import time

import cv2
import numpy as np
import torch
import pandas as pd

#https://stackoverflow.com/questions/60674501/how-to-make-black-background-in-cv2-puttext-with-python-opencv
def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

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


start_time = time.time()
images_path = 'inputs/detector_test/images'
images = []
for filename in os.listdir(images_path):
    img = cv2.imread(os.path.join(images_path, filename))
    # img = read_image(os.path.join(images_path, filename))
    images.append(img)

labels_path = 'inputs/detector_test/labels/'
labels_dfs = []
num_labels = 0
for filename in os.listdir(labels_path):
    df = pd.read_csv(os.path.join(labels_path, filename))
    labels_dfs.append(df)
    num_labels = num_labels + len(df)

# # yolov4
# read in the network from the saved config and weight files
net = cv2.dnn.readNetFromDarknet('yolo_config_files/yolov4.cfg', 'yolo_config_files/yolov4.weights')
# set the network to be a detection model
yolov4 = cv2.dnn_DetectionModel(net)
# set the image size and input params
yolov4.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
yolov4.setNmsAcrossClasses(True)
# #yolov5
# yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True, _verbose=False)
# yolov5.classes = [2, 3, 5, 7]
# yolov5.conf = 0.7
# yolov5.iou = 0.5
# SSD
# ssd = cv2.dnn.readNetFromCaffe('ssd_config_files/MobileNetSSD_deploy.prototxt', 'ssd_config_files/MobileNetSSD_deploy.caffemodel')
# CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",  "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# do detection on each image

model_detections = []

hws = []
for img in images:
    hws.append(img.shape[:2])
    detections = yolov4.detect(img, nmsThreshold=0.5, confThreshold=0.5)
    #detections = yolov5(img)
    # blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843,
    # (300, 300), 127.5)
    # ssd.setInput(blob)
    # detections = ssd.forward()
    # format the detections properly
    detections = get_output_format(detections)
    #detections = detections.pandas().xyxy[0]
    model_detections.append(detections)

# calculate iou
# iterate over each image gt labels
total_detections = 0
correct_detections = 0
true_class_count = 0
total_iou = 0
for labels, detections, img, path in zip(labels_dfs, model_detections, images, os.listdir(images_path)):
    # labels is all the label for a specific image
    # detections is all the detections for a specific image
    # idx = 0
    #for idx, detection in detections.iterrows():
    for detection in detections:
        # for i in np.arange(0, detections.shape[2]):
        detection_bbx = xywh2xyxy(detection[0])
        #detection_bbx = list([detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']])

        # text = str(detection_bbx)
        # cv2.putText(img, text, (int(detection_bbx[0]) - 20, int(detection_bbx[3])), cv2.FONT_HERSHEY_SIMPLEX,1,color=color,thickness=3)

        # confidence = detections[0, 0, i, 2]
        # cl = int(detections[0, 0, i, 1])
        # if confidence > 0 and cl in [6,7,14]:
        #     h, w = hw
        #     detection_bbx = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        #
        #

        best_iou = 0
        best_iou_class_match = 0
        for index, label in labels.iterrows():
            label_bbx = list([label['xmin'], label['ymin'], label['xmax'], label['ymax']])
            # do iou
            iou = bb_intersection_over_union(label_bbx, detection_bbx)
            # if its the best iou then save the value and class
            if iou >= best_iou:
                best_iou = iou
                #if str(detection['name']) == str(label['class']):
                if detection[2] == str(label['class']) and best_iou >0.5:
                    best_iou_class_match = 1
                else:
                    best_iou_class_match = 0

        # detections[idx] = list(detections[idx])
        # detections[idx].append(best_iou)
        # detections[idx].append(best_iou_class_match)
        # idx = idx + 1
        if best_iou > 0.5:
            correct_detections = correct_detections + 1
            true_class_count = true_class_count + best_iou_class_match
            total_iou = total_iou + best_iou
        else:
            correct_detections = correct_detections - 1
        total_detections = total_detections + 1

        if detection[2] == 'car':
            color = (9,127, 240)
        if detection[2] == 'bus':
            color = (54, 41, 159)
        if detection[2] == 'truck':
            color = (124, 88, 27)
        if detection[2] == 'motorbike':
            color = (66, 133, 78)
        cv2.rectangle(img, (int(detection_bbx[0]), int(detection_bbx[1])),
                      (int(detection_bbx[2]), int(detection_bbx[3])), color=color, thickness=3)
        text = '{}: {}'.format(detection[2], round(float(detection[1]),2))
        draw_text(img, text,text_color=(255,255,255), text_color_bg=color, pos=(int(detection_bbx[0]), int(detection_bbx[1]-25)))
        cv2.imwrite('inputs/detector_test/images_out/{}'.format(path), img)
# for detections in model_detections:
#     total_detections = total_detections + len(detections)
#     for idx, detection in detections.iterrows():
#         true_class_count = true_class_count + detection['best_class_match']
#         #total_iou = total_iou + detection[3]

class_accuracy = true_class_count / total_detections
iou_accuracy = total_iou / total_detections
count_accuracy = correct_detections / num_labels

end_time = time.time()
elapsed_time = round(end_time - start_time, 2)
print('class accuracy:', round(class_accuracy, 2))
print('iou accuracy:', round(iou_accuracy, 2))
print('count accuracy:', round(count_accuracy, 2))
print('Execution time:  ', elapsed_time, 'seconds')
