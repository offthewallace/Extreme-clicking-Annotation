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

cv2.imshow("edges", im)
cv2.imshow("edgeboxes", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()