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

xmin = min(point1x,point2x,point3x,point4x)
xmax =max(point1x,point2x,point3x,point4x)
ymin=min(point1y,point2y,point3y,point4y)
ymax=max(point1y,point2y,point3y,point4y)

extremPoint=[]
#img2=img[][]
img = img[ymin:ymax,xmin:xmax]
(ROW,COL,_)= img.shape
print(ROW,COL)
detector = cv2.ximgproc.createStructuredEdgeDetection('model.yml.gz')
rgb_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)





class cell:
	def __init__(self, x, y, distance):
		self.x = x
		self.y=y		
		self.distance=distance
	def __eq__(self, other):

		if self.x == other.x and self.y==other.y and self.distance==other.distance:
			return True
		else:
			return False


def isInsideGrid(i, j): 

	return (i >= 0 and i < COL and j >= 0 and j < ROW) 

def printPath(parent, startY,startX,i,j,img): 
	
	#Base Case : If j is source 
	#if parent[i][j][1] == -1: 
	#	print (i,j)
	#	return
	#printPath(parent,parent[i][j][0],parent[i][j][1]) 
	#print (i,j) 
	while not i==-1 or not j ==-1:
		print(i,j)
		img[i][j]=(0, 255, 0)
		i = parent[i][j][0]
		j=parent[i][j][1] 
	return img
# Method returns minimum cost to reach bottom 
# right from top left 
def shortest(grid, row, col,StartRow,StartCol,desRow,desCol):


	dis= [[float("Inf") for x in range(col)] for x in range(row)]
	#print(dis)
	parent = [[[] for x in range(col)] for x in range(row)]

	for x in range(row):
		for y in range(col):
			parent[x][y].append(-1)
			parent[x][y].append(-1)
	#print(parent)
	#print(parent)
	# direction arrays for simplification of getting  neighbour 
	dx = [-1, 0, 1, 0] 
	dy = [0, 1, 0, -1]

	#set<cell> st; 
	st = [] 

	# insert (0, 0) cell with 0 distance 
	st.append(cell(StartRow, StartCol, 0))
	#initialize distance of (0, 0) with its grid value 
	dis[StartRow][StartCol] = grid[StartRow][StartCol]

	#loop for standard dijkstra's algorithm 
	while st:
	
		# get the cell with minimum distance and delete 
		# it from the set 
		k = st.pop()
		#st.erase(st.begin()); 
		print("k.x is",k.x)
		print('k.y is ',k.y)
		if k.x == desRow and k.y == desCol:
					print('done')
					break

		# looping through all neighbours 
		for i in range(4): 
			x = k.x + dx[i] 
			y = k.y + dy[i] 
			# if not inside boundry, ignore them 
			if not isInsideGrid(x, y): 
				continue

			# If distance from current cell is smaller, then 
			# update distance of neighbour cell 
			if dis[x][y] > dis[k.x][k.y] + grid[x][y]:
				# If cell is already there in set, then 
				# remove its previous entry 
				
				if dis[x][y] != float("Inf"):
					st.remove(cell(x, y, dis[x][y])) 

				# update the distance and insert new updated 
				# cell in set 
				#print('update')
				dis[x][y] = dis[k.x][k.y] + grid[x][y] 
				print("x is",x)
				print('y is ',y)
				st.append(cell(x, y, dis[x][y]))

				st.sort(key=lambda x: x.distance, reverse=True)
				parent[x][y]=[]
				parent[x][y].append(k.x)
				parent[x][y].append(k.y)
				print('parent[x][y] is',parent[x][y])
				
	#printPath(parent, desRow,desCol)
	#print(dis)

	return dis[row - 1][col - 1], parent
im = detector.detectEdges(np.float32(rgb_im) / 255.0)
orimap = detector.computeOrientation(im)
edges = detector.edgesNms(im, orimap)

Matrix = edges.tolist()
print((Matrix[ROW-1][COL-1]))

for x in range(len(edges.tolist())):
	for y in range(len(edges.tolist()[0])):
		Matrix[x][y]=1-Matrix[x][y]
		if Matrix[x][y] ==1:
			Matrix[x][y]+=5

print(point1y-ymin,point1x-xmin,point2y-ymin,point2x-xmin)
dist,parent=shortest(Matrix, ROW, COL,point1y-ymin,point1x-xmin,point2y-ymin,point2x-xmin)
print(point1y-ymin,point1x-xmin,point2y-ymin,point2x-xmin)
print(parent[253][1])
#print(point2y-ymin,point2x-xmin)
#print(parent[point2y-ymin][point2x-xmin])


img = printPath(parent, point1y-ymin,point1x-xmin,point2y-ymin,point2x-xmin,img)
print(dist)
cv2.imwrite('left.jpg',img)


#cv2.imshow("edges", im)
#cv2.imshow("edgeboxes", edges)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

'''

<point1x>309</point1x>
<point1y>76</point1y>
<point2x>56</point2x>
<point2y>149</point2y>
<point3x>367</point3x>
<point3y>635</point3y>
<point4x>617</point4x>
<point4y>571</point4y>
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
#rect = (max(0,xmin-25),max(0,ymin-25),xmax-25,ymax-25)
rect=(xmin,ymin,xmax,ymax)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
#img = cv2.cv2tColor(img, cv2.COLOR_BGR2RGBA)
return img
'''
