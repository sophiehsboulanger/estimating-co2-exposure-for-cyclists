import os
import cv2


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

# read in the network from the saved config and weight files
net = cv2.dnn.readNetFromDarknet('yolo_config_files/yolov4.cfg', 'yolo_config_files/yolov4.weights')
# set the network to be a detection model
yolov4 = cv2.dnn_DetectionModel(net)
# set the image size and input params
yolov4.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)