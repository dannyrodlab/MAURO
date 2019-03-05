from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2 as cv2
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from skimage.morphology import watershed

# Initialize variables
filename = 'BSDS_small/train/22090.jpg'
colordos = 'lab'
method = 'kmeans'
n_clusters = 3

## Colour Spaces

def moveColor(image, color):
    # choices=['rgb','lab','hsv','rgb+xy','lab+xy','hsv+xy']
    if color == 'rgb':
        #rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgbImage = image
        return rgbImage
    if color == 'hsv':
        hsvImage = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsvImage = cv2.normalize(hsvImage, None, alpha = 0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return hsvImage   
    if color == 'lab':
        labImage = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        labImage = cv2.normalize(labImage, None, alpha = 0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return labImage
    if color == 'gray':
        grayImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return grayImage
#   TODO
#    if color == 'rgb+xy':
#    if color == 'lab+xy':
#    if color == 'hsv+xy':
    return 0

## Clustering
 
def clusterImage(image, method):
    # choices=['kmeans','gmm',hierarchical','watershed']
    #scikit-learn expects 2D num arrays for the training set for a fit function.
    #stackoverflow/questions/34972142
    prediction = []
    train = image.reshape((nx*ny,channels)) # vectors must live in RGB=R^3.
    if method == 'kmeans':
        kmeans = KMeans(n_clusters)
        kmeans.fit(train)
        prediction = kmeans.labels_.astype(np.int).reshape(nx,ny)
    if method == 'gmm':
        gmm = GaussianMixture(n_clusters)
        gmm.fit(train)
        prediction = gmm.predict(train).reshape(nx,ny)
    if method == 'hierarchical':
        hcl = AgglomerativeClustering(n_clusters)
        hcl.fit(train)
        prediction = hcl.labels_.astype(np.int).reshape(nx,ny)
#   TODO
#    if method == 'watershed':
#        return watershed
    return prediction

#def buildPrediction(labels,[x,y,z]):
#    if hasattr(,'labels_'):
#        prediction = 
#    return

def groundtruth():
    import scipy.io as sio
    gth = sio.loadmat(filename.replace('jpg','mat'))
    seg = gth['groundTruth'][0,5][0][0]['Segmentation']
    return seg

## Debugging

image = imageio.imread(filename)
nx, ny, channels = image.shape
rgbImage = moveColor(image,'rgb')
hsvImage = moveColor(image,'hsv')
labImage = moveColor(image,'lab')
grayImage = moveColor(image,'gray')

figure = plt.figure(1)
axis = figure.add_subplot(3,1,1)
plt.imshow(labImage)
axis = figure.add_subplot(3,1,2)
plt.imshow(hsvImage)
axis = figure.add_subplot(3,1,3)
plt.imshow(grayImage,cmap='gray')
plt.show()

kmeans = clusterImage(image,'kmeans')
#Al final no recupero los centroides; unicamente las clases.
#kmeans_centers = kmeans.cluster_centers_
#kmeans_labels = kmeans.labels_
gmm = clusterImage(image,'gmm')
# No memory for HCL :(
#hcl = clusterImage(image,'hierarchical')
segmentation = groundtruth()

figure = plt.figure(2)
axis = figure.add_subplot(3,1,1)
plt.imshow(kmeans)
axis = figure.add_subplot(3,1,2)
plt.imshow(gmm)
axis = figure.add_subplot(3,1,3)
plt.imshow(segmentation)
plt.show()

def segmentByClustering( rgbImage, colorSpace, clusteringMethod, numberOfClusters):
    return segmentation
