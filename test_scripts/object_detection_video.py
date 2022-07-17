import cv2


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()


video = cv2.VideoCapture('../inputs/test_video_bike_stab_2.mp4')
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
FRAME_SKIP = 1
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print_progress_bar(0, total_frames, prefix='Progress:', suffix='Complete')
while video.isOpened():
    ret, frame = video.read()

    # if frame is read correctly ret is True
    if not ret:
        print('Cant receive frame (stream end?). Exiting...')
        break
    if frame_no % FRAME_SKIP == 0:  # only do tracking and detection every FRAME_SKIP frames
        classIds, scores, boxes = model.detect(frame, confThreshold=0.7, nmsThreshold=0.4)  # , nmsThreshold=0.4

        # put bounding boxes, score and class id on image
        for (classId, score, box) in zip(classIds, scores, boxes):
            # bounding box
            cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                          color=(0, 255, 0), thickness=3)

            # text
            # text = '%s: %.2f' % (classes[classId], score)
            text = str(score)
            cv2.putText(frame, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color=(0, 255, 0), thickness=2)
    frames.append(frame)
    print_progress_bar(frame_no, total_frames, prefix='Progress:', suffix='Complete')
    frame_no = frame_no + 1

video.release()
# get the first image in the array to get size params
img = frames[0]
height, width, layers = img.shape
# choose codec according to format needed
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('../outputs/test_video_bike_output_7.mp4', fourcc, 25, (width, height))
for frame in frames:
    video.write(frame)
video.release()
