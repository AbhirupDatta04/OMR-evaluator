import cv2 as cv
import numpy as np
import imutils
import argparse
from argparse import ArgumentParse
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
        
questionCnts = contours.sort_contours(questionCnts, method = "top-to-bottom")[0]
correct = 0


for(q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
   
    cnts = contours.sort_contours(questionCnts[i:i+5])[0]
    bubbled = None
 

    for (j, c) in enumerate(cnts):
        
        mask = np.zeros(thresh.shape, dtype = "uint8") #mask that reveals only the current "bubble" for the question
        cv.drawContours(mask, [c], -1, 255, -1)

     
        mask = cv.bitwise_and(thresh, thresh, mask = mask)# #mask application to the threshold image
        total = cv.countNonZero(mask)     # count the number of non-zero pixels in the bubble area

    
        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)


    color = (0, 0, 255)
    k = ANSWER_KEY[q]  #looking up for the answer in the answer key and initialize the contour color and the index of the correct answer

    #To see if the bubbled answer is correct
    if k == bubbled[1]:
        color = (0, 255, 0)
        correct += 1
        
        #to draw the outlineof the correct answer on the test
    cv.drawContours(paper, [cnts[k]], -1, color, 3)

score = (correct / 5.0) * 100
print ("[INFO] score: {:.2f}%".format(score))
cv.putText(paper, "{:.2f}%".format(score), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
cv.imshow("Original", image)
cv.imshow("Exam", paper)
cv.waitKey(0)
       