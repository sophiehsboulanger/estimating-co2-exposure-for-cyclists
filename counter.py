import sys
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import torch
import os.path


# get the output from model and put it in the correct format for object detector. A list of detections, each in tuples
# of ( [left,top,w,h], confidence, detection_class )
def get_output_format(frame_detections):
    # define output list
    output = []
    # define desired classes
    target_classes = [2, 3, 5, 7]  # is currently set to car, motorbike, bus and truck
    # unpack the tuple to get individual arrays
    class_ids, scores, boxes = frame_detections
    for (classId, score, box) in zip(class_ids, scores, boxes):
        if classId in target_classes:
            output.append((box, score, classId))
    return output


# Print iterations progress
# https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters?noredirect=1&lq=1
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


def main(max_age, min_age, nms_max_overlap, frame_skip, input_file, output, conf_threshold, nms_threshold):
    # check for gpu
    gpu = torch.cuda.is_available()

    # define the tracker
    tracker = DeepSort(max_age, nms_max_overlap, embedder_gpu=gpu)
    # tracker.tracker.n_init should be the minimum age before a track is confirmed
    tracker.tracker.n_init = min_age

    # open the class names files and then read them into a list
    with open('yolo_config_files/coco.names', 'r') as f:
        classes = f.read().splitlines()

    # read in the network from the saved config and weight files
    net = cv2.dnn.readNetFromDarknet('yolo_config_files/yolov4.cfg', 'yolo_config_files/yolov4.weights')

    # set the network to be a detection model
    object_detector = cv2.dnn_DetectionModel(net)

    # set the image size and input params
    object_detector.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

    # check if file exists
    if os.path.exists(input_file):
        # read in video
        video = cv2.VideoCapture(input_file)
    else:
        # if file doesn't exist, exit
        sys.exit('File does not exist')

    TOTAL_FRAMES = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    FRAME_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_number = 1
    frames = []
    tracks = []
    counted_vehicles = []

    # choose codec according to format needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output, fourcc, 25, (FRAME_WIDTH, FRAME_HEIGHT))

    print_progress_bar(0, TOTAL_FRAMES, prefix='Progress:', suffix='Complete')
    while video.isOpened():
        ret, frame = video.read()

        # if frame is read correctly ret is True
        if not ret:
            print('Cant receive frame, stream end. Exiting...')
            break
        if frame_number % frame_skip == 0:  # only do tracking and detection every FRAME_SKIP frames
            # do the detection on the frame and get in format needed for tracker
            detections = get_output_format(object_detector.detect(frame=frame, confThreshold=conf_threshold, nmsThreshold=nms_threshold))
            # track
            tracks = tracker.update_tracks(detections, frame=frame)
            # iterate through all the tracks to draw onto the image
        for track in tracks:
            # print('tracking')  # this is for debugging
            # get track id
            track_id = track.track_id
            if track_id not in counted_vehicles and track.state == 2:
                counted_vehicles.append(track_id)
            # get bounding box min x, min y, max x, max y
            bb = track.to_ltrb(orig=True)
            # if the track confirmed set the bounding box colour to green
            if track.state == 2:
                color = (0, 255, 0)
            # else set the colour to red
            else:
                color = (0, 0, 255)

            # draw bounding box
            cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])),
                          color=color, thickness=3)

            # get the detected class
            detected_class = track.get_det_class()
            # get the id of the track
            # text = 'track id: %s, class: %s' % (track_id, detected_class)
            text = track_id

            # put the text onto the image
            cv2.putText(frame, text, (int(bb[0]) - 20, int(bb[3])), cv2.FONT_HERSHEY_SIMPLEX, 1, color=color,
                        thickness=3)
            # print(track_id)
        # add the frame with the drawn on bounding boxes to the frame list
        # frames.append(frame)
        # save the frame to the output video
        output_video.write(frame)
        # print(frame_number)  # this is for debugging
        print_progress_bar(frame_number, TOTAL_FRAMES, prefix='Progress:', suffix='Complete')
        # increment the frame number
        frame_number = frame_number + 1

    video.release()
    print('Finished processing video')
    print('Counted vehicles: %d' % len(counted_vehicles))
    print(counted_vehicles)


if __name__ == "__main__":
    input_file = 'inputs/test_video_1.mp4'
    output_file = 'outputs/test.mp4'
    main(max_age=5, min_age=5, nms_max_overlap=0.5, frame_skip=5, input_file=input_file, output=output_file,
         conf_threshold=0.75, nms_threshold=0.4)
