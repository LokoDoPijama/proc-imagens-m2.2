from copy import deepcopy
import os
import numpy as np
import cv2
import sys
from random import randint as randi
from math import sqrt
import matplotlib.pyplot as plt

OutputPath = "output_seg1"

if not os.path.exists(OutputPath):
    os.makedirs(OutputPath)


def bgr2hex(bgr):
    return "#%02x%02x%02x" % (int(bgr[2]), int(bgr[1]), int(bgr[0]))

def ScatterPlot(img, centroids, clusterLabels, plotNameOut="scatterPlot.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            ax.scatter(img[x, y, 2], img[x, y, 1], img[x, y, 0], color = bgr2hex(centroids[clusterLabels[x, y]]))
    # Save before show(), since closing the window unregisters the figure.
    fig.savefig(os.path.join(OutputPath, plotNameOut))
    # plt.show()

def ShowCluster(img, centroids, clusterLabels, imgNameOut="out.png", XYMode=False):
    result = np.zeros((img.shape), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            bgr = centroids[clusterLabels[i, j]]
            # if clusterLabels[i, j] == 0:
            #     bgr = [0, 0, 0]
            # if clusterLabels[i, j] == 1:
            #     bgr = [255, 0, 0]
            # if clusterLabels[i, j] == 2:
            #     bgr = [0, 255, 0]
            # if clusterLabels[i, j] == 3:
            #     bgr = [0, 0, 255]
            
            result[i, j, 0] = np.uint8(bgr[0])
            result[i, j, 1] = np.uint8(bgr[1])
            result[i, j, 2] = np.uint8(bgr[2])
    cv2.imwrite(os.path.join(OutputPath, imgNameOut), result)
    if not XYMode:
        ScatterPlot(img, centroids, clusterLabels, plotNameOut="scatterPlot.png")
    # cv2.imshow("K-Mean Cluster", result)
    # cv2.waitKey(0)

def GetEuclideanDistance(Cbgr, Ibgr, x, y, XYmode=False):
    b = float(Cbgr[0]) - float(Ibgr[0])
    g = float(Cbgr[1]) - float(Ibgr[1])
    r = float(Cbgr[2]) - float(Ibgr[2])
    if XYmode:
        x_diff = float(Cbgr[3]) - float(x)
        y_diff = float(Cbgr[4]) - float(y)
        return sqrt(b*b + g*g + r*r + x_diff*x_diff + y_diff*y_diff)
    return sqrt(b*b + g*g + r*r)

def AreRGBCentroidsUnique(centroids):
    for i in range(centroids.shape[0]):
        for j in range(centroids.shape[0]):
            if np.array_equal(centroids[i, :3], centroids[j, :3]) and i != j:
                return False
    return True

def generateCentroids(k, img, XYmode=False):
    centroids = np.zeros((k, 3 if not XYmode else 5), dtype=np.float64)

    while AreRGBCentroidsUnique(centroids) != True:
            for i in range(k):
                #start get the initial point of segmentation in X and Y coordinates 
                x = randi(0, img.shape[0]-1)
                y = randi(0, img.shape[1]-1)
                #get the RGB (BGR) from img and divide in different matrix
                b = float(img[x, y, 0])
                g = float(img[x, y, 1])
                r = float(img[x, y, 2])
    
                centroids[i, 0] = b
                centroids[i, 1] = g
                centroids[i, 2] = r
                if (XYmode):
                    centroids[i, 3] = x
                    centroids[i, 4] = y
    
    return centroids

def KMeans(img, k=2, XYMode=False, max_iterations=100, imgNameOut="out.png", customCentroids=None):
    '''
    KMeans algorithm base to 3D data image (RGB)
    input: 
        img: image open using a library like OpenCV
        k: number of segments
        XYMode: flag to indicate if XY coordinates should be used
        max_iterations: number max of interations to segment image 
    '''
    Clusters = k
    
    # create the centroids of algorithm - each k is there is a center called "centroid"
    if customCentroids is None:
        centroids = generateCentroids(k, img, XYmode=XYMode)  #for 5D, create a matrix of zeros with (k, 5)
    else:
        centroids = customCentroids[:k, :3] if not XYMode else customCentroids[:k, :5]

    print("Centroids:\n", centroids)
    ClusterLabels = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(max_iterations):
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                MinDist = sys.float_info.max
                for c in range(Clusters):
                    dist = GetEuclideanDistance(centroids[c], img[x, y], x, y, XYmode=XYMode)
                    if dist <= MinDist:
                        MinDist = dist
                        ClusterLabels[x, y] = c

        # update Mean of the clusters
        if XYMode:
            MeanCluster = np.zeros((Clusters, 6), dtype=np.float64)
        else:
            MeanCluster = np.zeros((Clusters, 4), dtype=np.float64)

        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                clusterNumber = ClusterLabels[x, y]
                MeanCluster[clusterNumber, 0] += 1
                MeanCluster[clusterNumber, 1] += float(img[x, y, 0])
                MeanCluster[clusterNumber, 2] += float(img[x, y, 1])
                MeanCluster[clusterNumber, 3] += float(img[x, y, 2])
                if XYMode:
                    MeanCluster[clusterNumber, 4] += float(x)
                    MeanCluster[clusterNumber, 5] += float(y)
        
        copy = np.copy(centroids)
        for c in range(Clusters):
            # print("MeanCluster["+ str(c) +", 0]:", MeanCluster[c, 0])
            centroids[c, 0] = MeanCluster[c, 1] / MeanCluster[c, 0]
            centroids[c, 1] = MeanCluster[c, 2] / MeanCluster[c, 0]
            centroids[c, 2] = MeanCluster[c, 3] / MeanCluster[c, 0]
            if XYMode:
                centroids[c, 3] = MeanCluster[c, 4] / MeanCluster[c, 0]
                centroids[c, 4] = MeanCluster[c, 5] / MeanCluster[c, 0]

        Same = True
        for i in range(centroids.shape[0]):
            for j in range(centroids.shape[1]):
                if copy[i, j] != centroids[i, j]:
                    Same = False
                    break
            if not Same:
                break
        if Same:
            break
    ShowCluster(img, centroids, ClusterLabels, imgNameOut, XYMode)


ImageNames = ["2apples.jpg", "2or4objects.jpg", "colors.jpg"]
PrefixPath = "dataset/"
No = 1
Image = cv2.imread(PrefixPath + ImageNames[No])
Image = cv2.resize(Image, None, fx=0.25, fy=0.25)
print("Image Size:", Image.shape)

customCentroids = generateCentroids(5, Image, XYmode=True) # Mesmos centroids em todos os testes para garantir consistência

KMeans(Image, k=2+1, XYMode=False, max_iterations=10, imgNameOut="img_out_3D_k3.png", customCentroids=deepcopy(customCentroids))
# KMeans(Image, k=4+1, XYMode=False, max_iterations=10, imgNameOut="img_out_3D_k5.png", customCentroids=deepcopy(customCentroids))
# KMeans(Image, k=2+1, XYMode=True, max_iterations=10, imgNameOut="img_out_5D_k3.png", customCentroids=deepcopy(customCentroids))
# KMeans(Image, k=4+1, XYMode=True, max_iterations=10, imgNameOut="img_out_5D_k5.png", customCentroids=deepcopy(customCentroids))

