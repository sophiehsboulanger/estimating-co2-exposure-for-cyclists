import sys
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import torch
import os.path
import vehicle_distance
import Vehicle


def get_bb_area(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]

    width = x2 - x1
    height = y2 - y1

    return abs(width * height)


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
        # only pass bounding boxes if they are a target class
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


def main(input_file, output, max_age=5, min_age=5, nms_max_overlap=0.5, frame_skip=5, conf_threshold=0.75,
         nms_threshold=0.4, min_size=12800):
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

    # set up the vehicle distance calculator
    vdc_setup_img = cv2.imread('inputs/straight_2m.png')
    vdc = vehicle_distance.VehicleDistanceCalculator(vdc_setup_img, 2000)

    # check if input video file exists
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
    vehicles = {}  # track_ids, vehicle object
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
            detections = get_output_format(
                object_detector.detect(frame=frame, confThreshold=conf_threshold, nmsThreshold=nms_threshold))
            if len(detections) > 0:
                # track
                tracks = tracker.update_tracks(detections, frame=frame)
        # iterate through all the tracks
        for track in tracks:
            # get track id
            track_id = track.track_id
            # get bounding box min x, min y, max x, max y
            bb = track.to_ltrb(orig=True)
            # if the track id is not already in the list, create a vehicle object and add it to the dictionary
            if track_id not in vehicles.keys():
                # create vehicle object
                vehicle = Vehicle.Vehicle(track_id, track.get_det_class, bb)
                # add it to the dictionary
                vehicles[track_id] = vehicle
            # if the track confirmed and above the minimum area then continue
            if track.state == 2 and get_bb_area(bb) >= min_size:
                # get the vehicle object associated to the track
                vehicle = vehicles[track_id]
                # confirm the count
                vehicle.confirmed = True
                # get the cropped vehicle image
                vehicle_img = frame[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])]
                # placeholder text
                distance_text = 'no distance'
                # check there is an image
                if len(vehicle_img) > 0:
                    # try and find a licence plate
                    vehicle_lp = vdc.find_licence_plate(vehicle_img)
                    if vehicle_lp is not None:
                        # if a lp is found try and calculate the distance
                        distance = vdc.distance_to_licence_plate(vehicle_lp)
                        if distance is not None:
                            # if a distance is found, put it in meters (2dp)
                            distance_text = round(distance / 1000, 2)
                            # add the distance in mm to the vehicles distance list
                            vehicle.distances.append(distance)
                else:
                    # if there is no image print the track id, used for debugging
                    print(track_id)

                # set the colour of the bounding box to green
                color = (0, 255, 0)
                # draw bounding box
                cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])),
                              color=color, thickness=3)
                # format the text
                text = 'id: {}, distance: {}'.format(track_id, distance_text)
                # put the text onto the image
                cv2.putText(frame, text, (int(bb[0]) - 20, int(bb[3])), cv2.FONT_HERSHEY_SIMPLEX, 1, color=color,
                            thickness=3)

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
    print('Counted vehicles: %d' % len(vehicles))
    print(vehicles)
    # unique = set(counted_classes)
    # for counted_class in unique:
    #     class_count = counted_classes.count(counted_class)
    #     print("%s: %d" % (counted_class, class_count))

    return len(vehicles)


if __name__ == "__main__":
    input_file = 'inputs/test_video_bike_stab.mp4'
    output_file = 'outputs/code_update.mp4'
    main(input_file, output_file)
