# adapted from https://www.section.io/engineering-education/license-plate-detection-and-recognition-using-opencv-and-pytesseract/
# https://pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/
import cv2
import numpy as np
import torch


class VehicleDistanceCalculator:
    """This is a class to contain the vehicle distant calculator object

    focal_length is to be calculated later, known_height is the known height of a licence plate character in mm

    :param min_ar: the minimum aspect ratio of licence plate character, defaults to 0.55
    :type min_ar: int, optional

    :param max_ar: the maximum aspect ratio of the licence plate character, defaults to 0.75
    :type max_ar: int, optional

    :param debug: set depending on if you want the calculator to be in debug mode, defaults to `False`
    :type debug: boolean, optional

    :param accuracy: the number of decimal places to use in rounding operations, defaults to 2
    :type accuracy: int, optional


    """
    def __init__(self, min_ar=0.55, max_ar=0.75, debug=False, accuracy=2):
        """Constructor method
        """
        self.min_ar = min_ar
        self.max_ar = max_ar
        self.debug = debug
        self.accuracy = accuracy
        # focal length of the camera, to be calculated
        self.focal_length = None
        # height of the licence plate characters in mm
        self.known_height = 79

    def debug_imshow(self, img, title):
        """Function used to show processed images when in debug mode

        :param img: the image to show
        :type img: image
        :param title: the title to display in the output frame, can be used to identify images
        :type title: string
        """
        cv2.imshow(title, img)
        if self.debug:
            cv2.waitKey(0)

    def find_licence_plate(self, img):
        lp_detector = torch.hub.load('C:/Users/sophi/PycharmProjects/licencePlateDetector/yolov5', 'custom',
                                     path='C:/Users/sophi/PycharmProjects/licencePlateDetector/yolov5/runs/train/exp7/weights/best.pt',
                                     source='local')  # default

        lp_detector.conf = 0.6

        # detect the licence plate, should only be one
        results = lp_detector(img)
        # results.show()
        results = results.xyxy[0].cpu().numpy()
        # print(results)
        for result in results:
            # get xmin, ymin, xmax, ymax of the bb for the licence plate
            # xyxy = results.xyxy[0].cpu().numpy()
            # xyxy = xyxy[0]
            xmin, ymin, xmax, ymax, *rest = np.rint(result).astype('int')
            ar = round((xmax - xmin) / float((ymax - ymin)), 2)
            if 4 <= ar <= 6:
                # return the crop of the licence plate
                return img[ymin:ymax, xmin:xmax]

    def get_average_character_height(self, lp):
        heights = []
        # convert to grey
        img = cv2.cvtColor(lp, cv2.COLOR_BGR2GRAY)
        # denoise
        img = cv2.fastNlMeansDenoising(img, h=10)
        # detect edges
        edged = cv2.Canny(img, 20, 200)
        # find contours
        contours, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
        for contour in contours:
            # get the bounding box
            x, y, w, h = cv2.boundingRect(contour)
            # calculate the aspect ratio
            ar = round(w / float(h), 2)
            # print(ar)
            # img2 = img.copy()
            # cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # cv2.putText(img2, str(ar), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 255, 0), thickness=1, )
            # cv2.imshow("fdsf", img2)
            # cv2.waitKey(0)
            # if it's between the accepted aspect ratios
            if self.max_ar >= ar >= self.min_ar:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                # save h in heights
                heights.append(h)
        cv2.imshow("fdsf", img)
        cv2.waitKey(0)
        # return the average height of the characters and exit, otherwise try again
        return round(sum(heights) / len(heights), self.accuracy)

    def distance_to_camera(self, per_height):
        # workout and return distance from object to camera (per_height = perceived height)
        return (self.known_height * self.focal_length) / per_height

    def calculate_focal_length(self, lp, known_distance):
        avg_height = self.get_average_character_height(lp)
        # focal length = (measured height in pixels * known distance in mm) / known height in mm
        focal_length = (avg_height * known_distance) / self.known_height
        self.focal_length = focal_length
        print(focal_length)
        return focal_length


if __name__ == "__main__":
    vdc = VehicleDistanceCalculator()
    # ----------- set the focal length -----------
    # read in the image
    img = cv2.imread("inputs/straight_2m.png")
    # find the licence plate
    lp = vdc.find_licence_plate(img)
    # calculate and set the focal length
    vdc.calculate_focal_length(lp, 2000)

    # -------- now test it works on another image ----------
    img = cv2.imread("inputs/straight_4m.png")
    lp = vdc.find_licence_plate(img)
    if lp is not None:
        # get the perceived height of the lp characters
        height = vdc.get_average_character_height(lp)
        # calculate the distance
        distance = vdc.distance_to_camera(height)
        print(distance)
    else:
        print('no licence plate detected')
