import math
import numpy as np
import sys, os
import matplotlib.pyplot as plt
from lxml import etree as ET
import cv2
import random 
import pdb
#ROW=640
#COL=640

img = cv2.imread('messi5.jpg')
#print(path)
point1x=309
point1y=76
point2x=56
point2y=149
point3x=367
point3y=635
point4x=617
point4y=571


(ROW,COL,_)= img.shape
print(ROW,COL)
detector = cv2.ximgproc.createStructuredEdgeDetection('model.yml.gz')
rgb_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
im = detector.detectEdges(np.float32(rgb_im) / 255.0)
orimap = detector.computeOrientation(im)
edges = detector.edgesNms(im, orimap)
for x in range(len(edges.tolist())):
	for y in range(len(edges.tolist()[0])):
		im[x][y]=1-im[x][y]
		edges[x][y]=1-edges[x][y]
		#if Matrix[x][y] ==1:
		#	Matrix[x][y]=1000
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
	#rect = (max(0,xmin-25),max(0,ymin-25),xmax-25,ymax-25)
rect=(xmin,ymin,xmax,ymax)
cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)

print(im.shape)
cv2.imshow("edges", im)
cv2.imshow("edgeboxes", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()