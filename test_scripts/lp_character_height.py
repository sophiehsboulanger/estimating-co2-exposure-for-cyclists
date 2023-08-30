# adapted from https://www.section.io/engineering-education/license-plate-detection-and-recognition-using-opencv-and
# -pytesseract/
import cv2

debug = True
lp_min_ar = 4
lp_max_ar = 5
lt_min_ar = 0.5
lt_max_ar = 0.7


def debug_imshow(img, debug, title):
    cv2.imshow(title, img)
    if debug:
        cv2.waitKey(0)


def find_contours(img):
    # convert to grey
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    debug_imshow(img, debug, "grey")
    # denoise
    img = cv2.fastNlMeansDenoising(img)
    debug_imshow(img, debug, "denoise")

    # detect edges
    edged = cv2.Canny(img, 20, 200)
    debug_imshow(edged, debug, 'edges')

    # find contours
    contours, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image1 = img.copy()
    cv2.drawContours(image1, contours, -1, (0, 255, 0), 3)
    debug_imshow(image1, debug, 'all contours')

    # take the top 30 contours, based on area, and sort them (largest to smallest)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
    image2 = img.copy()
    cv2.drawContours(image2, contours, -1, (0, 255, 0), 3)
    debug_imshow(image2, debug, 'top 30 contours')
    return contours


# read in image
img = cv2.imread("../inputs/bus_licence_plate.png")
debug_imshow(img, debug, "original")

# iterate through the contours
for c in find_contours(img):
    # get the bounding box
    x, y, w, h = cv2.boundingRect(c)
    # calculate the aspect ratio
    ar = round(w / float(h))
    if lp_min_ar <= ar <= lp_max_ar:  # if it's between the accepted aspect ratios
        lp = img[y:y + h, x:x + w]
        debug_imshow(lp, debug, 'licence plate')
        # exit the loop as we've found the licence plate
        break

for c in find_contours(lp):
    # get the bounding box
    x, y, w, h = cv2.boundingRect(c)
    # calculate the aspect ratio
    ar = round(w / float(h), 2)
    if lt_min_ar <= ar <= lt_max_ar:  # if it's between the accepted aspect ratios
        cv2.rectangle(lp, (x, y), (x+w, y+h), color=(0, 255, 0), thickness= 3)
debug_imshow(lp, debug, 'letters')
