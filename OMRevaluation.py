import cv2 as cv
import numpy as np
import imutils
import argparse
from imutils.perspective import four_point_transform
from imutils import contours

#argument parser to parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path to the input image")
args = vars(ap.parse_args())

ANSWER_KEY = {0:1, 1:4, 2:0, 3:3, 4:1}

#edge detection
image = cv.imread(args["image"])
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray, (5,5), 0)
edged = cv.Canny(blurred, 75, 200)


cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
docCnt = None

#to find one contour was found
if len(cnts) > 0:
    #sort contours according to their size in decreasing order
    cnts = sorted(cnts, key = cv.contourArea, reverse = True)
    
    for c in cnts:
        #contour approximation
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4: #four points to assume we have the paper
            docCnt = approx
            break
        else:
            print ("[INFO] No contours detected.")

paper = four_point_transform(image, docCnt.reshape(4,2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))

#applying Otsu's thresholding method to binarize the warped piece of paper
thresh = cv.threshold(warped, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]

cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
questionCnts = []


for c in cnts:
    
    (x, y, w, h) = cv.boundingRect(c)
    ar = w / float(h)

    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        questionCnts.append(c)
        
        