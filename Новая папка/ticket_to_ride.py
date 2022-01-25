from typing import Union

from collections import defaultdict
from itertools import combinations

import numpy as np
import cv2
from skimage import img_as_float32, img_as_ubyte
from skimage.exposure import  equalize_hist
from skimage.measure import label
from skimage.transform import rescale
from skimage.morphology import remove_small_objects

COLORS = ('blue', 'green', 'black', 'yellow', 'red')
TRAINS2SCORE = {1: 1, 2: 2, 3: 4, 4: 7, 6: 15, 8: 21}


def tr_count(cln,color):
    cr_dict = {'blue': 133.,
              'green': 123.,
              'black': 153.,
              'yellow': 120.,
              'red': 123.}
    contours, hierarchy = cv2.findContours(cln, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    length = 0
    boxes = []
    for contour in contours[1:]:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int16(box)
        boxes.append(box)
        length1 = np.linalg.norm(box[0] - box[1])
        length2 = np.linalg.norm(box[1] - box[2])
        length += np.int16(max(length1, length2))
    n_trains = np.int8(length / cr_dict[color])
    return n_trains

def tr_colour_black(RGB):
    cln = (RGB[..., 0] > 0) & (RGB[..., 0] < 35) & (RGB[..., 1] > 0) & (RGB[..., 1] < 35) & (RGB[..., 2] > 0) & (
                RGB[..., 2] < 35)
    cln = remove_small_objects(cln, min_size=1500)
    cln_int = cln.astype(np.uint8)
    kernel = np.ones((3, 3))
    cln_int = cv2.morphologyEx(cln_int, cv2.MORPH_CLOSE, kernel)
    cln_int = cv2.morphologyEx(cln_int, cv2.MORPH_OPEN, kernel, iterations=1)
    cln_crop = cln_int[90:2490, 90:3730]
    cln = cln_crop
    return cln

def tr_colour_green(LAB):
    cln = (LAB[..., 1] < 108) & (LAB[..., 1] > 85) & (LAB[..., 2] > 60) & (LAB[..., 2] < 145)
    cln = remove_small_objects(cln, min_size=700)
    cln_int = cln.astype(np.uint8)
    cln = cln_int
    return cln

def tr_colour_yellow(LAB):
    cln = (LAB[..., 2] > 182) & (LAB[..., 2] < 202)
    cln = remove_small_objects(cln, min_size=800)

    cln_int = cln.astype(np.uint8)
    kernel = np.ones((7, 7))
    cln_int = cv2.morphologyEx(cln_int, cv2.MORPH_CLOSE, kernel, iterations=1)
    cln = cln_int
    return cln

def tr_colour_blue(Y_CR_CB):
    cln = (Y_CR_CB[..., 0] > 20) & (Y_CR_CB[..., 0] < 78) & (Y_CR_CB[..., 1] > 85) & (Y_CR_CB[..., 1] < 113) & (
                Y_CR_CB[..., 2] > 140) & (Y_CR_CB[..., 2] < 165)
    cln = remove_small_objects(cln, min_size=700)
    cln_int = cln.astype(np.uint8)
    kernel = np.ones((7, 7))
    cln_int = cv2.morphologyEx(cln_int, cv2.MORPH_CLOSE, kernel, iterations=1)
    kernel = np.ones((5, 5))
    cln_int = cv2.morphologyEx(cln_int, cv2.MORPH_OPEN, kernel, iterations=1)
    cln_crop = cln_int[90:2490, 90:3750]
    cln = cln_crop

    return cln

def tr_colour_red(RGB, Y_CR_CB):
    cln = (Y_CR_CB[..., 1] > 190) & (Y_CR_CB[..., 1] < 208)
    img2 = RGB.copy()

    cln = remove_small_objects(cln, min_size=700)

    cln_int = cln.astype(np.uint8)
    cln = cln_int

    return cln

def find_cntrs(img):
    img_temp = img.copy()[..., ::-1]
    img_gray_temp = cv2.cvtColor(img_temp, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = img_gray_temp[786:840, 1499:1553]
    w, h = template.shape[::-1]
    match = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.5
    lbl, n = label(match >= threshold, connectivity=2, return_num=True)
    lbl = np.int16([np.round(np.mean(np.argwhere(lbl == i), axis=0)) for i in range(1, n + 1)])
    centers = [[pt[0] + w // 2, pt[1] + h // 2] for pt in lbl]
    return centers


def scoress(cln,color):
    сr_dict = {'blue': 133.*40.,
              'green': 124.*40.,
              'black': 150.*65.,
              'yellow': 120.*40.,
              'red': 123.*40.}
    contours, _ = cv2.findContours(cln , cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    counts = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > сr_dict[color]:
            counts.append(np.int8(area / сr_dict[color]))
    scr = 0
    if len(counts):
        for a in counts:
            if a == 5:
                scr += 8
            elif a == 7:
                scr+= 16
            elif a > 8:
                n = a % 8
                if n == 5:
                    scr += 29
                elif n == 7:
                    scr += 37
            else:
                if a == 1:
                    scr += 1
                elif a == 2:
                    scr += 2
                elif a == 3:
                    scr += 4
                elif a == 4:
                    scr += 7
                elif a == 6:
                    scr += 15
                elif a == 8:
                    scr += 21
    return scr


def predict_image(img: np.ndarray) -> (Union[np.ndarray, list], dict, dict):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    centers = find_cntrs(img)
    img = img[..., ::-1]
    LAB = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    Y_CR_CB = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    RGB = img

    bl = tr_colour_black(RGB)
    blU = tr_colour_blue(Y_CR_CB)
    re = tr_colour_red(RGB, Y_CR_CB)
    ye = tr_colour_yellow(LAB)
    gre = tr_colour_green(Y_CR_CB)

    black = tr_count(bl,'black')
    blue = tr_count(blU,'blue')
    red = tr_count(re,'red')
    yellow = tr_count(ye,'yellow')
    green = tr_count(gre,'green')

    scr_black = scoress(bl,'black')
    scr_blue = scoress(blU,'blue')
    scr_red = scoress(re,'red')
    scr_yellow = scoress(ye,'yellow')
    scr_green = scoress(gre,'green')

    if black <= 5: black = 0
    if blue <= 5: blue = 0
    if red <= 5: red = 0
    if yellow <= 5: yellow = 0
    if green <= 5: green = 0

    if scr_black <= 5: scr_black = 0
    if scr_blue <= 5: scr_blue = 0
    if scr_red <= 5: scr_red = 0
    if scr_yellow <= 5: scr_yellow = 0
    if scr_green <= 5: scr_green = 0

    n_trains = {'blue': blue, 'green': green,'black': black,  'yellow': yellow, 'red': red}
    scores = {"blue": scr_blue, "green": scr_green, "black": scr_black, "yellow": scr_yellow, "red": scr_red}
    # raise NotImplementedError
    # city_centers = np.int64([[1000, 2000], [1500, 3000], [1204, 3251]])
    # n_trains = {'blue': 20, 'green': 30, 'black': 0, 'yellow': 30, 'red': 0}
    # scores = {'blue': 60, 'green': 90, 'black': 0, 'yellow': 45, 'red': 0}
    return centers, n_trains, scores
# img = cv2.imread('Data/all.jpg')
# a,b,c=predict_image(img)
# print(a)
# print(b)
# print(c)