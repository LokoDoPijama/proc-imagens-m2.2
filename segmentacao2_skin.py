import os
import numpy as np
import cv2
from tqdm import tqdm
from queue import Queue

OutputPath = "output_seg2_skin"

if not os.path.exists(OutputPath):
    os.makedirs(OutputPath)


# Functions For Skin Based Thresholding 
# def RGB_Threshold(bgr):
#     b = float(bgr[0])
#     g = float(bgr[1])
#     r = float(bgr[2])

#     E1 = ? # add eqquaintion in paper 
#     E2 = ? # add eqquaintion in paper 

#     return E1 or E2

# def YCrCb_Threshold(yCrCb):
#     y = float(yCrCb[0])
#     Cr = float(yCrCb[1])
#     Cb = float(yCrCb[2])

#     E1 = ? # add eqquaintion in paper 
#     E2 = ? # add eqquaintion in paper 
#     E3 = ? # add eqquaintion in paper 
#     E4 = ? # add eqquaintion in paper 
#     E5 = ? # add eqquaintion in paper 

#     return E1 and E2 and E3 and E4 and E5

# def HSV_Threshold(hsv):
#     return ? ? # add eqquaintion in paper 

def Threshold(bgra, hsv, yCrCb):
    b = float(bgra[0])
    g = float(bgra[1])
    r = float(bgra[2])
    a = float(bgra[3])
    h = (float(hsv[0])/100) * 360 # Normalizado para 0-360
    s = float(hsv[1]) / 100 # Normalizado para 0-1
    v = float(hsv[2]) / 100 # Normalizado para 0-1
    y = float(yCrCb[0])
    Cr = float(yCrCb[1])
    Cb = float(yCrCb[2])

    # print(bgra, hsv, yCrCb)

    E1 = 0 <= h and h <= 50 and 0.23 <= s <= 0.68 and r > 95 and g > 40 and b > 20 and r > g and r > b and abs(r - g) > 15 and a > 15

    E2 = r > 95 and g > 40 and b > 20 and r > g and r > b and abs(r - g) > 15 and a > 15 and Cr > 135 and Cb > 85 and y > 80 and Cr <= (1.5862*Cb)+20 and Cr >= (0.3448*Cb)+76.2069 and Cr >= (-4.5652*Cb)+234.5652 and Cr <= (-1.15*Cb)+301.75 and Cr <= (-2.2857*Cb)+432.85

    return E1 or E2

# https://arxiv.org/ftp/arxiv/papers/1708/1708.02694.pdf
def SkinSegmentation(img, imgNameOut="out.png"):
    result = np.copy(img)
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    hsv = cv2.normalize(hsv, None, 0.0, 100.0, cv2.NORM_MINMAX, cv2.CV_32FC3)
    yCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    print(yCrCb.shape)

    for i in tqdm(range(img.shape[0])):
        for j in range(img.shape[1]):
            if (not Threshold(bgra[i, j], hsv[i, j], yCrCb[i, j])):
                result[i, j, 0] = 0
                result[i, j, 1] = 0
                result[i, j, 2] = 0

    cv2.imwrite(os.path.join(OutputPath, imgNameOut), result)

PrefixPath = "dataset/"
ImageList = ["face1.jpg", "face2.jpg", "face3.jpg", "face4.jpg", "colors.jpg"]

img = cv2.imread(PrefixPath + ImageList[2])
print(img.shape)

SkinSegmentation(img, "out_"+ImageList[2])

# for i in range(len(ImageList)):
#     img = cv2.imread(PrefixPath + ImageList[i])
#     print(img.shape)
#     SkinSegmentation(img, "out_"+ImageList[i])