from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
# taken from: realpython.com/python-opencv-color-spaces/#visualizing-nemo-in-rgb-color-space
# taken from: pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
r,g,b = cv2.split(rgbImage)
figure = plt.figure(1)
axis = figure.add_subplot(1,1,1, projection = '3d')
pixel_colors = rgbImage.reshape((np.shape(rgbImage)[0]*np.shape(rgbImage)[1],3))
norm = colors.Normalize(vmin=-1.,vmax=1)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

axis.scatter(r.flatten(),g.flatten(),b.flatten(),facecolors=pixel_colors,marker='.')
plt.show()

h,s,v = cv2.split(hsvImage)
figure = plt.figure(2)
axis = figure.add_subplot(1,1,1, projection = '3d')
axis.scatter(r.flatten(),g.flatten(),b.flatten(),facecolors=pixel_colors,marker='.')
plt.show()

# BGR -> GRAY
#cv2.imshow('segmentation', gray)
#cv2.waitKey()
# BGR -> HSV
#cv2.imshow('segmentation', hsv)
#cv2.waitKey()
#plt.hist(segmentation.ravel(),256,[0,256])
#plt.show()

