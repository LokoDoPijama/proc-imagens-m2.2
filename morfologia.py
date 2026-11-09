from copy import deepcopy
import math
import os
import numpy as np
import cv2
import sys
from random import randint as randi
from math import sqrt

# ================================ Funções de segmentação ================================

def SaveColoredClusters(img, centroids, clusterLabels, imgNameOut="out.png"):
    result = np.zeros((img.shape), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            bgr = centroids[clusterLabels[i, j]]
            result[i, j, 0] = np.uint8(bgr[0])
            result[i, j, 1] = np.uint8(bgr[1])
            result[i, j, 2] = np.uint8(bgr[2])
    cv2.imwrite(os.path.join(OutputPath, imgNameOut), result)

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

def GetKMeansClusterLabelsAndCentroids(img, k=2, XYMode=False, max_iterations=100, customCentroids=None):
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

    return ClusterLabels, centroids

def getClustersAsMasks(ClusterLabels, k):
    masks = []
    for c in range(k):
        mask = np.zeros(ClusterLabels.shape, dtype=np.uint8)
        for i in range(ClusterLabels.shape[0]):
            for j in range(ClusterLabels.shape[1]):
                if ClusterLabels[i, j] == c:
                    mask[i, j] = 255

        masks.append(mask)
    return masks

def getImageClustersAsMasksAndCentroids(img, k, XYMode=False, max_iterations=100, customCentroids=None):
    ClusterLabels, centroids = GetKMeansClusterLabelsAndCentroids(img, k, XYMode, max_iterations, customCentroids)
    masks = getClustersAsMasks(ClusterLabels, k)
    return masks, centroids

# ================================ Funções para morfologia ================================

def criarElementoEstruturante(tamanho=3, cruz=False):
    elEstruturante = np.full((tamanho, tamanho), 255, dtype=np.uint8)
    if cruz:
        elEstruturante[::2, ::2] = 0

    return elEstruturante

def sobreporElementoEstruturante(img, elEstruturante, i, j):
    tam = elEstruturante.shape[0]
    offsetPadrao = math.floor(tam/2)

    for x in range(tam):
        for y in range(tam):
            if elEstruturante[x, y] == 255:
                if img[i + x - offsetPadrao, j + y - offsetPadrao] == 0:
                    return False
    return True

def erosao(img, tamanhoElementoEstruturante=3, cruz=False):
    elEstrut = criarElementoEstruturante(tamanhoElementoEstruturante, cruz)
    tam = elEstrut.shape[0]
    offsetPadrao = tam // 2

    img_output = np.zeros_like(img)

    for i in range(offsetPadrao, img.shape[0] - offsetPadrao):
        for j in range(offsetPadrao, img.shape[1] - offsetPadrao):
            if sobreporElementoEstruturante(img, elEstrut, i, j):
                img_output[i, j] = 255
            else:
                img_output[i, j] = 0

    return img_output

def dilatacao(img, tamanhoElementoEstruturante=3, cruz=False):
    elEstrut = criarElementoEstruturante(tamanhoElementoEstruturante, cruz)
    tam = elEstrut.shape[0]
    offset = tam // 2

    img_output = deepcopy(img)

    for i in range(offset, img.shape[0] - offset):
        for j in range(offset, img.shape[1] - offset):
            if img[i, j] == 255:
                for x in range(tam):
                    for y in range(tam):
                        if elEstrut[x, y] == 255:
                            ni = i + x - offset
                            nj = j + y - offset

                            img_output[ni, nj] = 255

    return img_output

def abertura_morfologica(imagem, tamanho=3, cruz=False):
    """
    Realiza a abertura morfológica de uma imagem.

    Parâmetros:
        imagem: imagem de entrada em escala de cinza ou binária.
        tamanho: tamanho do elemento estruturante (inteiro positivo).
        cruz: Determina se o objeto estruturante será cruz ou quadrado.

    Retorno:
        Imagem após a operação de abertura.
    """
    if tamanho < 1:
        raise ValueError("O tamanho deve ser maior ou igual a 1.")

    imagem_erodida = erosao(imagem, tamanho, cruz)
    imagem_aberta = dilatacao(imagem_erodida, tamanho, cruz)

    return imagem_aberta

def fechamento_morfologico(imagem, tamanho=3, cruz=False):
    """
    Realiza o fechamento morfológico de uma imagem.

    Parâmetros:
        imagem: imagem de entrada em escala de cinza ou binária.
        tamanho: tamanho do elemento estruturante.
        cruz: Determina se o objeto estruturante será cruz ou quadrado.

    Retorno:
        Imagem após a operação de fechamento.
    """
    if tamanho < 1:
        raise ValueError("O tamanho deve ser maior ou igual a 1.")

    imagem_aberta = dilatacao(imagem, tamanho, cruz)
    imagem_fechada = erosao(imagem_aberta, tamanho, cruz)

    return imagem_fechada


# =============================== Outras ================================

def generateColorsForClusters(k):
    base_value = 255
    if (k > 6):
        base_value = base_value / (math.ceil(k/6) + 1)

    colors = []

    for m in range(0, k, 6):
        base_multiplier = m // 6 + 1
        colors.append([base_value*base_multiplier, 0, 0])
        colors.append([0, base_value*base_multiplier, 0])
        colors.append([0, 0, base_value*base_multiplier])
        colors.append([base_value*base_multiplier, base_value*base_multiplier, 0])
        colors.append([0, base_value*base_multiplier, base_value*base_multiplier])
        colors.append([base_value*base_multiplier, 0, base_value*base_multiplier])

    return colors

def getColorsFromCentroids(centroids):
    k = centroids.__len__()

    colors = np.zeros((k,3), dtype=np.uint8)

    for c in range(k):
        colors[c, 0] = np.uint8(centroids[c, 0])
        colors[c, 1] = np.uint8(centroids[c, 1])
        colors[c, 2] = np.uint8(centroids[c, 2])

    return colors

def getColoredClustersImageFromMasks(masks, colors):
    k = masks.__len__()
    result = np.zeros((masks[0].shape[0], masks[0].shape[1], 3), dtype=np.uint8)

    for i in range(masks[0].shape[0]):
        for j in range(masks[0].shape[1]):
            for m in range(k):
                if masks[m][i, j] == 255:
                    bgr = colors[m]
                    result[i, j, 0] = np.uint8(bgr[0])
                    result[i, j, 1] = np.uint8(bgr[1])
                    result[i, j, 2] = np.uint8(bgr[2])

    return result

def pipeline(Image, k=2, XYMode=False, max_iterations=10, customCentroids=None):
    masks, centroids = getImageClustersAsMasksAndCentroids(Image, k=k, XYMode=XYMode, max_iterations=max_iterations, customCentroids=deepcopy(customCentroids))

    colors = getColorsFromCentroids(centroids)

    kMeansClusters = getColoredClustersImageFromMasks(masks, colors)

    cv2.imwrite(os.path.join(OutputPath, "original_kmean_clusters.png"), kMeansClusters)

    new_masks = []

    for m in range(k):
        cv2.imwrite(os.path.join(OutputPath, f"input_mask_{m}.png"), masks[m])
        new_masks.append(abertura_morfologica(masks[m], tamanho=3, cruz=False))
        new_masks[m] = fechamento_morfologico(new_masks[m], tamanho=3, cruz=False)
        cv2.imwrite(os.path.join(OutputPath, f"output_mask_{m}.png"), new_masks[m])


    kMeansClusters = getColoredClustersImageFromMasks(new_masks, colors)

    cv2.imwrite(os.path.join(OutputPath, "final_result.png"), kMeansClusters)


# ================================ Uso das funções ================================

OutputPath = "output_morfologia"

if not os.path.exists(OutputPath):
    os.makedirs(OutputPath)

ImageNames = ["2apples.jpg", "2or4objects.jpg", "colors.jpg"]
PrefixPath = "dataset/"
No = 1
Image = cv2.imread(PrefixPath + ImageNames[No])
Image = cv2.resize(Image, None, fx=0.25, fy=0.25)
print("Image Size:", Image.shape)

customCentroids = generateCentroids(5, Image, XYmode=True) # Mesmos centroids em todos os testes para garantir consistência

pipeline(Image, k=2+1, XYMode=False, max_iterations=10, customCentroids=deepcopy(customCentroids))
# pipeline(Image, k=4+1, XYMode=False, max_iterations=10, customCentroids=deepcopy(customCentroids))
# pipeline(Image, k=2+1, XYMode=True, max_iterations=10, customCentroids=deepcopy(customCentroids))
# pipeline(Image, k=4+1, XYMode=True, max_iterations=10, customCentroids=deepcopy(customCentroids))