import cv2

# read in the image
img = cv2.imread("../inputs/image5.png")
img = cv2.resize(img, (960, 540))

# open the class names files and then read them into a list
with open('../yolo_config_files/coco.names', 'r') as f:
    classes = f.read().splitlines()

# read in the network from the saved config and weigh files
net = cv2.dnn.readNetFromDarknet('../yolo_config_files/yolov4.cfg', '../yolo_config_files/yolov4.weights')

# net the network to be a detection model
model = cv2.dnn_DetectionModel(net)

# set the image size input params
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

# do detection
classIds, scores, boxes = model.detect(img, confThreshold=0.80, nmsThreshold=0.4)
print(classIds)

# put bounding boxes, score and class id on image
for (classId, score, box) in zip(classIds, scores, boxes):
    # bounding box
    cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                  color=(0, 255, 0), thickness=3)

    # text
    text = '%s: %.2f' % (classes[classId], score)
    cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                color=(0, 255, 0), thickness=2)

# get unique values in the detected classes
detected_classes = []
for classId in classIds:
    if classId not in detected_classes:
        detected_classes.append(classId)

print('The detected classes and their counts for this image are:')
# change classIds to a list
classIds = classIds.tolist()
classes_counts = []
for classId in detected_classes:
    count = classIds.count(classId)
    classes_counts.append(count)
    print('%s: %d' % (classes[classId], count))

# save image
cv2.imwrite('../demos/object_detection5.png', img)
