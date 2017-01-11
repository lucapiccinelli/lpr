import cv2
import numpy as np
import random
import pytesseract
from PIL import Image

global im
global low_canny
global high_canny
global trackbars
global imgsnames
global img_counter

low_canny = 231
high_canny = 89
trackbars = False
imgsnames = [ \
    r'targa0.jpg', \
    r'targa1.jpg', \
    r'targa2.jpg', \
    r'targa3.jpg', \
    r'targa4.jpg', \
    r'targa5.jpg', \
    r'targa6.jpg',
    ]
img_counter = 0


def setImgsNames(x):
    global img_counter
    img_counter = x
    do_img_proc()


def setLowCannyTrack(x):
    global low_canny
    low_canny = x
    do_img_proc()


def setHighCannyTrack(x):
    global high_canny
    high_canny = x
    do_img_proc()


def do_img_proc():
    global im
    global trackbars

    im = cv2.imread(imgsnames[img_counter], cv2.IMREAD_GRAYSCALE)
    im_in = cv2.resize(im, dsize=(640, 480))
    np.array(im_in, copy=True)

    avg, sdv = cv2.meanStdDev(im_in)

    if avg < 55 and sdv < 40:
        im_in = cv2.equalizeHist(im_in)
        im_in = cv2.GaussianBlur(im_in, (3, 3), 1)

    im_in = cv2.GaussianBlur(im_in, (3, 3), 1)

    im_canny = cv2.Canny(im_in, low_canny, high_canny, apertureSize=3)
    im_canny_copy = np.array(im_canny, copy=True)

    h, w = im_in.shape
    im_out = np.zeros((h, w, 1), np.uint8)

    totaleArea =  float(w*h)

    contours, hierarchy = cv2.findContours(im_canny_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for (i,c) in enumerate(contours):
        rect = cv2.boundingRect(c)
        area = (rect[2] * rect[3]) / totaleArea
        hPerc =  rect[3] / float(h)
        aspect = float(rect[2]) / rect[3]

        if area < 0.002 or area > 0.015 or hPerc < 0.07 or aspect >= 0.7:
            continue

        #print area
        color = (255, 255, 255)

        mask = np.zeros((h, w, 1), np.uint8)
        cv2.rectangle(mask, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), color, -1)
        avg, sdv = cv2.meanStdDev(im_in, mask=mask)

        print avg, sdv

        offset = -random.randint(10, 20) if i % 2 == 1 else rect[3] + random.randint(15, 25)

        cv2.rectangle(im_out, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), color, -1)
        # cv2.putText(im_out, '{:0.2f}'.format(sdv[0, 0]), (rect[0], rect[1] + offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # cv2.drawContours(im_out, contours, i, color, -1)

    mask = np.array(im_out, copy=True)
    im_out = cv2.bitwise_and(im_in, im_out)
    im_out = cv2.bitwise_not(im_out, mask=mask)
    cv2.imwrite(r'test.jpg', im_out)

    avg = cv2.mean(im_out, mask)
    print avg

    t, im_out = cv2.threshold(im_out, avg[0] + 10, 255, cv2.cv.CV_THRESH_BINARY)

    cv2.imshow('canny', im_canny)
    if not trackbars:
        cv2.createTrackbar('imgnames', 'canny', img_counter, 6, setImgsNames)
        cv2.createTrackbar('canny_low', 'canny', low_canny, 255, setLowCannyTrack)
        cv2.createTrackbar('canny_high', 'canny', high_canny, 255, setHighCannyTrack)
        trackbars = True

    # im_out = cv2.morphologyEx(im_out, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.cv.CV_SHAPE_RECT, (3, 3)), iterations=1)

    cv2.imshow('im', im_out)
    cv2.imshow('out', im_in)


    print pytesseract.image_to_string(Image.fromarray(im_out), config='-psm 7')

do_img_proc()
cv2.waitKey(0)
cv2.destroyAllWindows()

#
# kernel = np.ones((3, 3), np.uint8)
# im_out = cv2.morphologyEx(im_out, cv2.MORPH_DILATE, kernel, anchor=(2, 0), iterations=1)
#
# im_out = cv2.bitwise_not(im_out)

#
# t, im_in = cv2.threshold(im, 200, 255, cv2.THRESH_BINARY_INV)
