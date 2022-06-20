import cv2

video = cv2.VideoCapture('../inputs/test_video_3.mp4')
# open the class names files and then read them into a list
with open('../yolo_config_files/coco.names', 'r') as f:
    classes = f.read().splitlines()

# read in the network from the saved config and weigh files
net = cv2.dnn.readNetFromDarknet('../yolo_config_files/yolov4.cfg', '../yolo_config_files/yolov4.weights')

# net the network to be a detection model
model = cv2.dnn_DetectionModel(net)

# set the image size input params
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

frames = []
frame_no = 1
while video.isOpened():
    ret, frame = video.read()

    # if frame is read correctly ret is True
    if not ret:
        print('Cant receive frame (stream end?). Exiting...')
        break
    classIds, scores, boxes = model.detect(frame, confThreshold=0.7, nmsThreshold=0.4)

    # put bounding boxes, score and class id on image
    for (classId, score, box) in zip(classIds, scores, boxes):
        # bounding box
        cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                      color=(0, 255, 0), thickness=3)

        # text
        text = '%s: %.2f' % (classes[classId], score)
        cv2.putText(frame, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color=(0, 255, 0), thickness=2)
    frames.append(frame)
    print(frame_no)
    frame_no = frame_no + 1
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
# get the first image in the array to get size params
img = frames[0]
height, width, layers = img.shape
# choose codec according to format needed
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('../outputs/test_video_3_output.mp4', fourcc, 25, (width, height))
for frame in frames:
    video.write(frame)
video.release()
