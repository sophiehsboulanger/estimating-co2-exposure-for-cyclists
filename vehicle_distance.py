# adapted from https://www.section.io/engineering-education/license-plate-detection-and-recognition-using-opencv-and-pytesseract/
# https://pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/
import cv2
import numpy as np
from roboflow import Roboflow


class VehicleDistanceCalculator:
    """This is a class to contain the vehicle distant calculator object

    focal_length is to be calculated later, known_height is the known height of a licence plate character in mm

    :param img: image of a vehicle with a licence plate

    :param distance: known distance from the camera to the license plate in mm
    :type distance: int

    :param min_ar: the minimum aspect ratio of licence plate character, defaults to 0.55
    :type min_ar: int, optional

    :param max_ar: the maximum aspect ratio of the licence plate character, defaults to 0.75
    :type max_ar: int, optional

    :param debug: set depending on if you want the calculator to be in debug mode, defaults to `False`
    :type debug: boolean, optional

    """

    def __init__(self, img, distance, min_ar=0.55, max_ar=0.75, debug=False):
        """Constructor method
        """
        # lp detector
        rf = Roboflow(api_key="gefD2gnKVfDlmkoHBp96")
        project = rf.workspace().project("license-plate-recognition-rxg4e")
        self.lp_detector = project.version(3).model
        self.lp_detector.confidence = 0.6

        self.min_ar = min_ar
        self.max_ar = max_ar
        self.debug = debug
        # height of the licence plate characters in mm
        self.known_height = 79
        # focal length of the camera using image and known distance
        lp = self.find_licence_plate(img)
        self.focal_length = self.calculate_focal_length(lp, distance)

    def debug_imshow(self, title, img, scale=1):
        """Function used to show processed images when in debug mode

        :param scale: the scale by which the image is to be displayed, default 1
        :param img: the image to show
        :type img: image
        :param title: the title to display in the output frame
        :type title: string
        """
        if self.debug:
            new_size = tuple(i * scale for i in img.shape[0:2])
            new_size = new_size[::-1]
            #img = cv2.resize(img, new_size)
            cv2.namedWindow(title)  # Create a named window
            cv2.moveWindow(title, 40, 30)
            cv2.imshow(title, img)
            cv2.waitKey(0)

    def find_licence_plate(self, img):
        """finds the licence plate in the image

        designed for single vehicle use so only returns one licence plate

        :param img: the image of the vehicle
        :return: the section of img that contains the licence plate, or None if no licence plate is found
        """

        # detect the licence plate, should only be one
        results = self.lp_detector.predict(img).json()
        for result in results['predictions']:
            # get x1, x2, y1, y2 of the bb for the licence plate
            # example box object from the Pillow library
            x1 = int(result['x'] - result['width'] / 2)
            x2 = int(result['x'] + result['width'] / 2)
            y1 = int(result['y'] - result['height'] / 2)
            y2 = int(result['y'] + result['height'] / 2)
            ar = round(result['width'] / result['height'], 2)
            # self.debug_imshow('found licence plate', img[y1:y2, x1:x2])
            # print(ar)

            if 2 <= ar <= 6:
                self.debug_imshow('selected licence plate', img[y1:y2, x1:x2], 10)
                # return the crop of the licence plate
                cv2.imwrite('inputs/distance_test_imgs/lp_og.png', img[y1:y2, x1:x2])
                return img[y1:y2, x1:x2]
        return None

    def get_average_character_height(self, lp):
        """calculates the average height in pixels

        performs edge and contour detection to locate shapes within the licence plate
        then uses the known aspect ratio of a licence plate character to filter results to give licence plate characters
        then calculates and returns the average height in pixels

        :param lp: an image of a licence plate
        :return: the average height in pixels of the licence plate characters
        :type: float
        """
        heights = []
        # convert to grey
        img = cv2.cvtColor(lp, cv2.COLOR_BGR2GRAY)
        # denoise
        img = cv2.fastNlMeansDenoising(img, h=10)
        # threshold
        ret, img = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
        # erode
        kernel = np.ones((2, 1), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)
        # find contours
        self.debug_imshow('enhanced licence plate', img, 10)
        contours, new = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
        # get the area of the image of the lp
        lp_a = img.shape[0] * img.shape[1]
        # calculate the area threshold of the characters based on the lp area
        a_thresh = int(lp_a / 25)
        for contour in contours:
            # get the bounding box
            x, y, w, h = cv2.boundingRect(contour)
            # calculate the aspect ratio
            ar = round(w / float(h), 2)
            # calculate the area
            a = w * h
            # if it's between the accepted aspect ratios
            if self.max_ar >= ar >= self.min_ar:
                if a >= a_thresh:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    # save h in heights
                    heights.append(h)
        self.debug_imshow("found characters", img, 10)
        # if characters found, return the average height of the characters
        if len(heights) > 0:
            return sum(heights) / len(heights)
        else:
            return None

    def distance_to_licence_plate(self, lp):
        """calculates the distance to the licence plate

        uses the average height of the licence plate characters in pixels and the calculated focal length to
        determine the distance from the camera to the licence plate

        distance (mm) =
            (known height of lp characters (mm) * calculated focal length (px)) / perceived height of lp characters (px)

        :param lp: image of a licence plate
        :return: distance from the camera to the licence plate in mm
        """
        # get the average lp character height
        per_height = self.get_average_character_height(lp)
        # if characters are detected
        if per_height is not None:
            # workout and return distance from object to camera (per_height = perceived height)
            return (self.known_height * self.focal_length) / per_height
        else:
            return None

    def calculate_focal_length(self, lp, known_distance):
        """calculates the focal length of the camera

        uses the average height of the licence plate characters in pixels and a known provided distance in mm

        focal length (px) =
            (perceived height of lp characters (px) * known distance (mm)) / known height of lp characters (mm)

        :param lp: the image of the licence plate
        :param known_distance: the known distance of the camera to the licence plate in mm
        :return: the focal length of the camera in px
        """
        avg_height = self.get_average_character_height(lp)
        # focal length = (measured height in pixels * known distance in mm) / known height in mm
        focal_length = (avg_height * known_distance) / self.known_height
        self.focal_length = focal_length
        return focal_length


if __name__ == "__main__":
    vdc_setup_img = cv2.imread('inputs/straight_2m.png')
    vdc = VehicleDistanceCalculator(vdc_setup_img, 2000, debug=True)

    # -------- now test it works on another image ----------
    img = cv2.imread("inputs/straight_3m.png")
    vehicle_lp = vdc.find_licence_plate(img)
    if vehicle_lp is not None:
        # if a lp is found try and calculate the distance
        distance = vdc.distance_to_licence_plate(vehicle_lp)
        if distance is not None:
            # if a distance is found, put it in meters (2dp)
            distance = round(distance / 1000, 2)
            print(distance)
        else:
            print('no distance found')
    else:
        print('no licence plate found')
