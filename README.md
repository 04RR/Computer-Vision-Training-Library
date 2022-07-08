# Subpixel - A Computer Vision Training Library 
Train Computer Vision models on your dataset with just a few lines of code. 

(Work In Progress)

'''
import requests
import cv2
import numpy as np
import serial

url = "http://192.168.0.218:8080//shot.jpg"
ser = serial.Serial("COM4", 9800, timeout=1)

x, y = 480, 270
area = x * y

red_thres = 1.0
blue_thres = 15.0


def get_circles(img):

    if len(img.shape) >= 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    minDist = 1000
    param1 = 500
    param2 = 200
    minRadius = 2
    maxRadius = 3

    circles = cv2.HoughCircles(
        img, cv2.HOUGH_GRADIENT, 1, minDist, param1, param2, minRadius, maxRadius
    )

    return circles


def find_blue(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([101, 50, 38])
    upper_blue = np.array([110, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    return mask


def find_red(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)

    return mask


def if_impeller(frame):
    img = cv2.resize(img, (x, y))

    mask_red = find_red(img)
    mask_blue = find_blue(img)

    circles_red = get_circles(mask_red)
    circles_blue = get_circles(mask_blue)

    if ((np.sum(mask_red) / 255.0) * 100 / area > red_thres) and (
        (np.sum(mask_blue) / 255.0) * 100 / area > blue_thres
    ):
        if (circles_blue is not None) and (circles_red is not None):
            return 1.
        else:
            return 0.


while True:

    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    img = cv2.resize(img, (x, y))

    mask_red = find_red(img)
    mask_blue = find_blue(img)

    circles_red = get_circles(mask_red)
    circles_blue = get_circles(mask_blue)

    if ((np.sum(mask_red) / 255.0) * 100 / area > red_thres) and (
        (np.sum(mask_blue) / 255.0) * 100 / area > blue_thres
    ):
        if (circles_blue is not None) and (circles_red is not None):
            print("There is an Impleller.")
            ser.write(b"H")
            ser.write(b"L")

    if cv2.waitKey(1) == 27:
        break

ser.close()
cv2.destroyAllWindows()
'''
