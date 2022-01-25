from typing import Union
import numpy as np
import cv2
from skimage.morphology import remove_small_objects

COLORS = ('blue', 'green', 'black', 'yellow', 'red')
TRAINS2SCORE = {1: 1, 2: 2, 3: 4, 4: 7, 6: 15, 8: 21}


def tr_count(cln,color):
    cr_dict = {'blue': 133., 'green': 121., 'black': 155., 'yellow': 109., 'red': 124.}
    contours, hierarchy = cv2.findContours(cln, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    leng = 0
    boxes = []
    for contour in contours[1:]:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int16(box)
        boxes.append(box)
        len1 = np.linalg.norm(box[0] - box[1])
        len2 = np.linalg.norm(box[1] - box[2])
        leng += np.int16(max(len1, len2))
    n_trains = np.int8(leng / cr_dict[color])
    return n_trains

def tr_colour_black(RGB):
    cln = (RGB[..., 0] > 0) & (RGB[..., 0] < 36) & (RGB[..., 1] > 0) & (RGB[..., 1] < 36) & (RGB[..., 2] > 0) & (
                RGB[..., 2] < 36)
    cln = remove_small_objects(cln, min_size=1500)
    cln_int = cln.astype(np.uint8)
    kernel = np.ones((3, 3))
    cln_int = cv2.morphologyEx(cln_int, cv2.MORPH_CLOSE, kernel, iterations=1)
    cln_int = cv2.morphologyEx(cln_int, cv2.MORPH_OPEN, kernel, iterations=1)
    cln_crop = cln_int[100:2480, 100:3720]
    cln = cln_crop

    return cln

def tr_colour_green(LAB):
    cln = (LAB[..., 1] < 108) & (LAB[..., 1] > 87) & (LAB[..., 2] > 65) & (LAB[..., 2] < 140)
    cln = remove_small_objects(cln, min_size=1500)
    cln_int = cln.astype(np.uint8)

    return cln_int

def tr_colour_yellow(Li_Am_B):
    cln = (Li_Am_B[..., 2] > 182) & (Li_Am_B[..., 2] < 198)
    cln = remove_small_objects(cln, min_size=1300)
    cln_int = cln.astype(np.uint8)
    kernel = np.ones((7, 7))
    cln_int = cv2.morphologyEx(cln_int, cv2.MORPH_CLOSE, kernel, iterations=1)

    return cln_int

def tr_colour_blue(Y_CR_CB):
    cln = (Y_CR_CB[..., 0] > 18) & (Y_CR_CB[..., 0] < 78) & (Y_CR_CB[..., 1] > 86) & (Y_CR_CB[..., 1] < 113) & (
                Y_CR_CB[..., 2] > 140) & (Y_CR_CB[..., 2] < 180)
    cln = remove_small_objects(cln, min_size=700)
    cln_int = cln.astype(np.uint8)
    kernel = np.ones((9, 9))
    cln_int = cv2.morphologyEx(cln_int, cv2.MORPH_CLOSE, kernel, iterations=1)
    kernel = np.ones((5, 5))
    cln_int = cv2.morphologyEx(cln_int, cv2.MORPH_OPEN, kernel, iterations=1)
    cln_crop = cln_int[100:2480, 100:3740]
    cln = cln_crop

    return cln

def tr_colour_red(Y_CR_CB):
    cln = (Y_CR_CB[..., 1] > 189) & (Y_CR_CB[..., 1] < 212)
    cln = remove_small_objects(cln, min_size=1500)
    cln_int = cln.astype(np.uint8)

    return cln_int

def find_cntrs(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    circle = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 150, param1=58, param2=32, minRadius=22, maxRadius=31)
    contours = None
    if circle is not None:
        circ = np.uint16(np.around(circle[0]))
        contours = np.array(circ[:, :-1].tolist())
        contours = contours[:, [1, 0]]
    return contours


def scrs(cln,color):
    сr_dict = {'blue': 130 * 10,'green': 138 * 14,'black': 174 * 19,'yellow': 109 * 13,'red': 123 * 17}
    contours, _ = cv2.findContours(cln , cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    count = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > сr_dict[color]:
            count.append(np.int8(area / сr_dict[color]))
    scr = 0
    for a in count:
        if a == 5:
            scr += 8
        elif a == 7:
            scr += 16
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
    centers = find_cntrs(img)
    img = img[..., ::-1]
    Li_Am_B = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    Y_CR_CB = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    RGB = img

    bl = tr_colour_black(RGB)
    blU = tr_colour_blue(Y_CR_CB)
    re = tr_colour_red(Y_CR_CB)
    ye = tr_colour_yellow(Li_Am_B)
    gre = tr_colour_green(Y_CR_CB)

    black = tr_count(bl,'black')
    blue = tr_count(blU,'blue')
    red = tr_count(re,'red')
    yellow = tr_count(ye,'yellow')
    green = tr_count(gre,'green')

    scr_black = scrs(bl,'black')
    scr_blue = scrs(blU,'blue')
    scr_red = scrs(re,'red')
    scr_yellow = scrs(ye,'yellow')
    scr_green = scrs(gre,'green')

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
# img = cv2.imread('Data/black_blue_green.jpg')
# img = cv2.imread('Data/black_red_yellow.jpg')
# img = cv2.imread('Data/red_green_blue_inaccurate.jpg')
# img = cv2.imread('Data/red_green_blue.jpg')
# a,b,c=predict_image(img)
# print(a)
# print(b)
# print(c)



# def tr_colour_black(img):
#     img = img.copy()
#     HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
#     SAT = HLS[:, :, 2]
#     LIGHT = HLS[:, :, 1]
#
#     cln = (SAT < 0.499) & (LIGHT < 0.0929)
#     cln = erosion(cln)
#     cln = remove_small_objects(cln, 128)
#     cln = dilation(cln)
#     cln = img_as_ubyte(cln.astype(np.float32))
#
#     return cln
#
# def tr_colour_green(img):
#     img = img.copy()
#     HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
#     HUE = HLS[:, :, 0]
#
#     cln = ((HUE > 138) & (HUE < 158))
#     cln = erosion(cln)
#     cln = remove_small_objects(cln, 128)
#     cln = dilation(cln)
#     cln = img_as_ubyte(cln.astype(np.float32))
#
#     return cln
#
# def tr_colour_yellow(img):
#     img = img.copy()
#     HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
#     SAT = HLS[:, :, 2]
#     HUE = HLS[:, :, 0]
#
#     cln = (SAT > 0.81) & ((HUE > 34) & (HUE < 61))
#     cln = erosion(cln)
#     cln = remove_small_objects(cln, 128)
#     cln = dilation(cln)
#     cln = img_as_ubyte(cln.astype(np.float32))
#
#     return cln
#
# def tr_colour_blue(img):
#     img = img.copy()
#     HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
#     SAT = HLS[:, :, 2]
#     HUE = HLS[:, :, 0]
#
#     cln = (SAT > 0.699) & ((HUE > 205) & (HUE < 216.4))
#     cln = erosion(cln)
#     cln = remove_small_objects(cln, 128)
#     cln = dilation(cln)
#     cln = img_as_ubyte(cln.astype(np.float32))
#
#     return cln
#
# def tr_colour_red(img):
#     img = img.copy()
#     HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
#     SAT = HLS[:, :, 2]
#     HUE = HLS[:, :, 0]
#
#     cln = ((HUE < 0.48) | (HUE > 295)) & (SAT > 0.68)
#     cln = erosion(cln)
#     cln = remove_small_objects(cln, 128)
#     cln = dilation(cln)
#     cln = img_as_ubyte(cln.astype(np.float32))
#
#     return cln
# def tr_count(cln):
#     contours, _ = cv2.findContours(cln, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     areas = [cv2.contourArea(cont) for cont in contours]
#
#     average_area = min(np.median(areas), 280)
#     high_bound = average_area + 72
#     low_bound = average_area - 72
#
#     cnt = 0
#     for area in areas:
#         if area <= high_bound and area >= low_bound:
#             cnt += 1
#         elif area > high_bound:
#             cnt += area // average_area
#             cof = (area / average_area - area // average_area)
#             if cof >= 0.5:
#                 cnt += 1
#     return cnt

# def scoress(cln):
#     contours, _ = cv2.findContours(cln, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     areas = [cv2.contourArea(cont) for cont in contours]
#     average_area = min(np.median(areas), 280)
#     high_bound = average_area + 72
#     low_bound = average_area - 72
#     scr = 0
#     for a in areas:
#         if a <= high_bound and a >= low_bound:
#             scr+=1
#         elif a > high_bound:
#             res= a / average_area
#             if(res>1) & (res<=2):
#                 scr+=2
#             elif(res>2) & (res<3.5):
#                 scr+=4
#             elif (res>=3.5) & (res<5.5):
#                 scr+=7
#             elif (res>=5.5) & (res<=7):
#                 scr+=15
#             elif res>=8:
#                 scr+=21
#             cof = (a / average_area - a // average_area)
#             if cof >= 0.5:
#                 scr += 1
#     return scr