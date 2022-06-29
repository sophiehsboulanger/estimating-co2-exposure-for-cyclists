from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2


# get the output from model and put it in the correct format for object detector. A list of detections, each in tuples
# of ( [left,top,w,h], confidence, detection_class )
def get_output_format(frame_detections):
    # define output list
    output = []
    # define desired classes
    target_classes = [1, 2, 3, 5, 7]  # is currently set to car, bicycle, truck, bus and motorbike
    # unpack the tuple to get individual arrays
    class_ids, scores, boxes = frame_detections
    for (classId, score, box) in zip(class_ids, scores, boxes):
        if classId in target_classes:
            output.append((box, score, classId))
    return output


# define the tracker
tracker = DeepSort(max_age=30, nn_budget=70, nms_max_overlap=0.5, embedder_gpu=False)

# open the class names files and then read them into a list
with open('yolo_config_files/coco.names', 'r') as f:
    classes = f.read().splitlines()

# read in the network from the saved config and weight files
net = cv2.dnn.readNetFromDarknet('yolo_config_files/yolov4.cfg', 'yolo_config_files/yolov4.weights')

# set the network to be a detection model
object_detector = cv2.dnn_DetectionModel(net)

# set the image size and input params
object_detector.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

# read in video
video = cv2.VideoCapture('inputs/test_video_bike_2.mp4')

FRAME_SKIP = 1
frame_number = 1
frames = []
tracks = []
while video.isOpened():
    ret, frame = video.read()

    # if frame is read correctly ret is True
    if not ret:
        print('Cant receive frame, stream end. Exiting...')
        break
    if frame_number % FRAME_SKIP == 0:  # only do tracking and detection every FRAME_SKIP frames
        # do the detection on the frame and get in format needed for tracker
        detections = get_output_format(object_detector.detect(frame=frame, confThreshold=0.8))
        # track
        tracks = tracker.update_tracks(detections, frame=frame)
        # iterate through all the tracks to draw onto the image
    for track in tracks:
        print('tracking')  # this is for debugging
        # get track id
        track_id = track.track_id
        # get bounding box min x, min y, max x, max y
        bb = track.to_ltrb(orig=True)
        # draw bounding box
        cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])),
                      color=(0, 255, 0), thickness=3)

        # get the detected class
        detected_class = track.get_det_class()
        # get the id of the track
        text = 'track id: %s, class: %s' % (track_id, detected_class)
        # put the text onto the image
        cv2.putText(frame, text, (int(bb[0]), int(bb[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color=(0, 255, 0), thickness=3)

    # add the frame with the drawn on bounding boxes to the frame list
    frames.append(frame)
    print(frame_number)  # this is for debugging
    # increment the frame number
    frame_number = frame_number + 1

video.release()
# get the first image in the array to get size params
img = frames[0]
height, width, layers = img.shape
# choose codec according to format needed
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('outputs/test_video_bike_output.mp4', fourcc, 25, (width, height))
for frame in frames:
    video.write(frame)
video.release()
