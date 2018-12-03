import numpy as np
import cv2 
from matplotlib import pyplot as plt
img = cv2.imread('messi5.jpg')
#
#imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#ret,thresh = cv2.threshold(imgray,127,255,0)
#im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#print(contours[4])
#print(contours[4].shape)

#im3 =cv2.drawContours(im, contours, -1, (0,255,0), 3)

#plt.imshow(im3),plt.colorbar(),plt.show()

'''
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (50,50,450,290)
mask = np.zeros(img.shape[:2],np.uint8)
#cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
print(mask.shape)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
plt.imshow(img),plt.colorbar(),plt.show()
'''

# newmask is the mask image I manually labelled
img = cv2.imread('label.jpg')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)	
#img =np.float32(img) / 255.0
#mask = np.zeros(img.shape[:2],np.uint8)
#print(img[557][449])
#mask2=np.argwhere(img ==[0,255,0])

(ROW,COL,_)= img.shape
contours =[]
for i in range(0,ROW):
    for j in range(0,COL):
        if img[i,j,:][0] >80 and img[i,j,:][0]<110 and img[i,j,:][1] >175 and img[i,j,:][1]<205 and img[i,j,:][2] >80 and img[i,j,:][2]<110:
            contours.append([i,j])
            #print("row is ", i)
            #print('col is ', j)
#mask2=np.asarray(mask2).T.tolist()
contours=np.array(contours)
print(contours.shape)
im3 =cv2.drawContours(img, [contours], 0, (0,255,0), 3)
plt.imshow(im3),plt.show()
'''
#print(mask2.shape)
#img[mask2] = [0,0,255]
#mask[img == [0,255,0]] = 0
#print(img.shape)
#plt.imshow(mask2),plt.colorbar(),plt.show()

# wherever it is marked white (sure foreground), change mask=1
# wherever it is marked black (sure background), change mask=0
#mask[img == [0,255,0] ] = 0
#mask[newmask == 255] = 1
#mask, bgdModel, fgdModel = cv.grabCut(img,mask,None,bgdModel,fgdModel,5,cv.GC_INIT_WITH_MASK)
#mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
#img = img*mask[:,:,np.newaxis]
'''