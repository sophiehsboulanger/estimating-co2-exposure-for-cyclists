import sys
import time

from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import torch
import os.path
import vehicle_distance
import Vehicle


# get the output from model and put it in the correct format for object detector. A list of detections, each in tuples
# of ( [left,top,w,h], confidence, detection_class )
# https://stackoverflow.com/questions/60674501/how-to-make-black-background-in-cv2-puttext-with-python-opencv
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


def no_distance_handler(distances):
    """ Handles missing distances.

    If for whatever reason a distance cannot be found for a vehicle at a given point this function handles this.
    If this list of distances is populated then it returns the last known distance
    If there are no distances then it returns 0

    Both values might need to be looked at, I might try a different imputation method to get the unknown distance
    The zero value might also need changing depending on future handling of distances

    :param distances: the list of distances associated to a vehicle
    :return: int, either the last value in distances or 0
    """
    if len(distances) > 0:
        return distances[-1]
    else:
        return 0


def create_txt_file(file_name, length, fps, frame_skip, conf, max_age, min_age, area, results):
    with open(file_name, 'w') as f:
        f.write('{} | video length: {} | {} fps \n'.format(file_name, round(length,0), fps))
        f.write('frame skip: {} | confidence: {} | max age: {} | min age: {} | area: {} \n'.format(frame_skip, conf,
                                                                                                   max_age, min_age,
                                                                                                   area))
        for key, value in results.items():
            f.write('{}: {} \n'.format(key, value))


def main(input_file, output, max_age=10, min_age=4, nms_max_overlap=1, frame_skip=6, conf_threshold=0.5,
         nms_threshold=0.5, min_size=0.2):
    # check for gpu
    gpu = torch.cuda.is_available()
    # define the tracker
    tracker = DeepSort(max_age, nms_max_overlap, embedder_gpu=gpu)
    # tracker.tracker.n_init should be the minimum age before a track is confirmed
    tracker.tracker.n_init = min_age

    # read in the network from the saved config and weight files
    net = cv2.dnn.readNetFromDarknet('yolo_config_files/yolov4.cfg', 'yolo_config_files/yolov4.weights')

    # set the network to be a detection model
    object_detector = cv2.dnn_DetectionModel(net)

    # set the image size and input params
    object_detector.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

    # set up the vehicle distance calculator
    vdc_setup_img = cv2.imread('inputs/distance_test_imgs/straight_2m.png')
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
    FPS = int(video.get(cv2.CAP_PROP_FPS))
    divisor = int(FPS / frame_skip)
    frame_number = 1
    vehicles = {}  # track_ids, vehicle object
    ids = []  # used to store ids currently in frame, using because track status doesn't delete properly
    # choose codec according to format needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output, fourcc, 25, (FRAME_WIDTH, FRAME_HEIGHT))

    # set up the progress bar
    print_progress_bar(0, TOTAL_FRAMES, prefix='Progress:', suffix='Complete')
    while video.isOpened():
        ret, frame = video.read()
        # if frame is read correctly ret is True
        if not ret:
            print('End of video. Exiting...')
            break
        if frame_number % frame_skip == 0:  # only do tracking and detection every FRAME_SKIP frames
            # do the detection on the frame and get in format needed for tracker
            detections = get_output_format(
                object_detector.detect(frame=frame, confThreshold=conf_threshold, nmsThreshold=nms_threshold))
            if len(detections) > 0:
                # track
                tracks = tracker.update_tracks(detections, frame=frame)
                # update vehicle list and vehicles in the list
                ids = []
                for track in tracks:
                    # get track id
                    track_id = track.track_id
                    ids.append(track_id)
                    # get bounding box min x, min y, max x, max y
                    bb = track.to_ltrb(orig=True)
                    # if the track id is not already in the list, create a vehicle object and add it to the dictionary
                    if track_id not in vehicles.keys():
                        # create vehicle object
                        vehicle_id = Vehicle.Vehicle(track_id, track.get_det_class(), bb, track.state,
                                                     track.get_det_conf(), FRAME_WIDTH, FRAME_HEIGHT)
                        # add it to the dictionary
                        vehicles[track_id] = vehicle_id
                    else:
                        # otherwise get the vehicle from the list
                        vehicle_id = vehicles[track_id]
                        # update the bounding box and area
                        vehicle_id.set_bb(bb, FRAME_WIDTH, FRAME_HEIGHT)
                        # update the state
                        vehicle_id.status = track.state
                        # update class
                        vehicle_id.vehicle_class = track.get_det_class()
                        # update confidence
                        vehicle_id.conf = track.get_det_conf()
                    # if the vehicle is confirmed and above the minimum area try and find the distance
                    if vehicle_id.status == 2 and vehicle_id.bb_area >= min_size:
                        # update the vehicle to be confirmed
                        vehicle_id.confirmed = True
                        # check there's a valid bb
                        valid = all(i >= 0 for i in vehicle_id.bb)
                        if valid:
                            # get the cropped vehicle image
                            vehicle_img = frame[int(vehicle_id.bb[1]):int(vehicle_id.bb[3]),
                                          int(vehicle_id.bb[0]):int(vehicle_id.bb[2])]
                        else:
                            vehicle_img = []
                        # check there is an image
                        if len(vehicle_img) > 0:
                            # try and find a licence plate
                            vehicle_lp = vdc.find_licence_plate(vehicle_img)
                            if vehicle_lp is not None:
                                # if a lp is found try and calculate the distance
                                distance = vdc.distance_to_licence_plate(vehicle_lp)
                                if distance is not None:
                                    # print(distance)
                                    # add the distance in mm to the vehicles distance list
                                    vehicle_id.add_distance(distance)
                                else:
                                    # if no distance is found, use the previous distance
                                    vehicle_id.add_distance(no_distance_handler(vehicle_id.distances))
                            else:
                                # if there is no lp, use the previous distance
                                vehicle_id.add_distance(no_distance_handler(vehicle_id.distances))
                        else:
                            # if there is no image, use the previous distance
                            vehicle_id.add_distance(no_distance_handler(vehicle_id.distances))
        # draw bbs
        for track_id in vehicles:
            vehicle_id = vehicles[track_id]
            if track_id in ids and vehicle_id.confirmed and vehicle_id.conf is not None:
                # if track_id in ids:
                vehicle_class = vehicle_id.vehicle_class
                if vehicle_class == 2:
                    color = (9, 127, 240)
                if vehicle_class == 5:
                    color = (54, 41, 159)
                if vehicle_class == 7:
                    color = (124, 88, 27)
                if vehicle_class == 3:
                    color = (66, 133, 78)
                # draw bounding box
                cv2.rectangle(frame, (int(vehicle_id.bb[0]), int(vehicle_id.bb[1])),
                              (int(vehicle_id.bb[2]), int(vehicle_id.bb[3])),
                              color=color, thickness=3)

                if len(vehicle_id.distances) > 0:
                    distance = round(vehicle_id.distances[-1] / 1000, 2)
                else:
                    distance = 'no distance'
                text = 'ID:{}, distance:{}'.format(vehicle_id.track_id, distance)
                draw_text(frame, text, text_color=(255, 255, 255), text_color_bg=color,
                          pos=(int(vehicle_id.bb[0]), int(vehicle_id.bb[1] - 25)))

        # save the frame to the output video
        output_video.write(frame)
        # update progress bar
        print_progress_bar(frame_number, TOTAL_FRAMES, prefix='Progress:', suffix='Complete')
        # increment the frame number
        frame_number = frame_number + 1

    video.release()
    print('Finished processing video')
    # print('Counted vehicles: %d' % len(vehicles))
    total_score = 0
    counted = 0
    cars = 0
    buses = 0
    trucks = 0
    motorbikes = 0
    for vehicle_id in vehicles:
        # only print and analyse confirmed vehicles
        if vehicles[vehicle_id].confirmed:
            score = vehicles[vehicle_id].get_score(divisor)
            # print(score)
            total_score = total_score + score
            counted = counted + 1
            if vehicles[vehicle_id].vehicle_class == 2:
                # car
                cars = cars + 1
            elif vehicles[vehicle_id].vehicle_class == 3:
                # motorbike
                motorbikes = motorbikes + 1
            elif vehicles[vehicle_id].vehicle_class == 5:
                # bus
                buses = buses + 1
            elif vehicles[vehicle_id].vehicle_class == 7:
                # 'truck'
                trucks = trucks + 1
    results = {
        'score': total_score,
        'total count': counted,
        'cars': cars,
        'buses': buses,
        'trucks': trucks,
        'motorbikes': motorbikes
    }
    print(results)
    file_name = os.path.basename(input_file)
    file_name = file_name[:-4]+'.txt'
    file_path = 'demos/outputs/{}'.format(file_name)
    create_txt_file(file_path, (TOTAL_FRAMES/FPS), FPS, frame_skip, conf_threshold, max_age, min_age, min_size, results)
    return results


if __name__ == "__main__":
    input_file = 'demos/inputs/demo1.mp4'
    output_file = 'demos/outputs/demo1.mp4'
    start_time = time.time()
    main(input_file, output_file)
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)
    print('elapsed time: ', elapsed_time)
