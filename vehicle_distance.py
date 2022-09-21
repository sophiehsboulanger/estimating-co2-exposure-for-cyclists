# adapted from https://www.section.io/engineering-education/license-plate-detection-and-recognition-using-opencv-and
# -pytesseract/
import cv2


class VehicleDistanceCalculator:
    def __init__(self, lp_min_ar=4, lp_max_ar=5, lt_min_ar=0.5, lt_max_ar=0.7, debug=False, accuracy=2):
        self.lp_min_ar = lp_min_ar
        self.lp_max_ar = lp_max_ar
        self.lt_min_ar = lt_min_ar
        self.lt_max_ar = lt_max_ar
        self.debug = debug
        self.accuracy = accuracy

    def debug_imshow(self, img, title):
        cv2.imshow(title, img)
        if self.debug:
            cv2.waitKey(0)

    def find_contours(self, img):
        # convert to grey
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.debug_imshow(img, "grey")

        # denoise
        img = cv2.fastNlMeansDenoising(img, h=10)
        self.debug_imshow(img, "denoise")

        # detect edges
        edged = cv2.Canny(img, 20, 200)
        self.debug_imshow(edged, 'edges')

        # find contours
        contours, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        image1 = img.copy()
        cv2.drawContours(image1, contours, -1, (0, 255, 0), 3)
        self.debug_imshow(image1, 'all contours')

        # take the top 30 contours, based on area, and sort them (largest to smallest)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)#[:30]
        image2 = img.copy()
        cv2.drawContours(image2, contours, -1, (0, 255, 0), 3)
        self.debug_imshow(image2, 'top 30 contours')
        return contours

    def get_character_height(self, img):
        found = False
        # iterate through the contours
        for c in self.find_contours(img):
            # approximate bounding contour
            # perimeter = cv2.arcLength(c, True)
            # approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
            # # if the contour has four sides
            # if len(approx) == 4:
            # get the bounding box
            x, y, w, h = cv2.boundingRect(c)
            # calculate the aspect ratio
            ar = round(w / float(h))
            # if it's between the accepted aspect ratios
            if self.lp_min_ar <= ar <= self.lp_max_ar:
                found = True
                lp = img[y:y + h, x:x + w]
                self.debug_imshow(lp, 'licence plate')
                # exit the loop as we've found the licence plate
                break
        if not found:
            print("no licence plate found")
            return found
        elif found:
            found = False
            # variable to hold bounding boxes
            heights = []
            # now we've found the licence plate, isolate the characters within
            for c2 in self.find_contours(lp):
                # get the bounding box
                x, y, w, h = cv2.boundingRect(c2)
                # calculate the aspect ratio
                ar = round(w / float(h), 2)
                if self.lt_min_ar <= ar <= self.lt_max_ar:  # if it's between the accepted aspect ratios
                    found = True
                    cv2.rectangle(lp, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=3)
                    heights.append(h)
            if found:
                # show the result if in debug mode
                self.debug_imshow(lp, 'letters')
                # return the average height of the characters
                return round(sum(heights) / len(heights), self.accuracy)
            else:
                print("no characters in licence plate found")
                return found


if __name__ == "__main__":
    vdc = VehicleDistanceCalculator(debug=True)
    test_image = cv2.imread('inputs/castle_medow.jpeg')
    test_image = test_image[668:1911, 653:2223]
    print(vdc.get_character_height(test_image))
