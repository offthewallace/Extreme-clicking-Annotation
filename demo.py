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
point5x=500
point5y=315

xmin = min(point1x,point2x,point3x,point4x)
xmax =max(point1x,point2x,point3x,point4x)
ymin=min(point1y,point2y,point3y,point4y)
ymax=max(point1y,point2y,point3y,point4y)

img = img[ymin:ymax,xmin:xmax]
(ROW,COL,_)= img.shape
print(ROW,COL)

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


def notInsideGrid(i, j): 

	return (i < 0 or i >= ROW or j < 0 or j >= COL) 

def printPath(parent,i,j,img): 
 
	while True:
		print(i,j)
		img[i][j]=(0, 255, 0)
		i = parent[i][j][0]
		j=parent[i][j][1]
		if i ==-1 or j ==-1:
			break
	print('done by plot') 
	return img


# Method returns minimum cost to reach bottom 
# right from top left 
def shortest(grid, row, col,StartRow,StartCol):


	dis= [[float("Inf") for x in range(col)] for x in range(row)]
	#print(dis)
	parent = [[[] for x in range(col)] for x in range(row)]

	for x in range(row):
		for y in range(col):
			parent[x][y].append(-1)
			parent[x][y].append(-1)

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
		print("k.x is",k.x)
		print('k.y is ',k.y)

		# looping through all neighbours 
		for i in range(4): 
			x = k.x + dx[i] 
			y = k.y + dy[i] 
			print("x is",x)
			print('y is ',y)
			#if k.x == desRow and k.y == desCol:
			#	print('done')
             #   break

			# if not inside boundry, ignore them 
			if notInsideGrid(x, y): 
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
				
				st.append(cell(x, y, dis[x][y]))

				
				parent[x][y]=[]
				parent[x][y].append(k.x)
				parent[x][y].append(k.y)
				print('parent[x][y] is',parent[x][y])
		st.sort(key=lambda x: x.distance, reverse=True)
	#img = printPath(parent, desRow,desCol,img)
	#print(dis)

	return dis[row - 1][col - 1], parent,img


detector = cv2.ximgproc.createStructuredEdgeDetection('model.yml.gz')
rgb_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)	
im = detector.detectEdges(np.float32(rgb_im) / 255.0)
orimap = detector.computeOrientation(im)
edges = detector.edgesNms(im, orimap)

Matrix = edges.tolist()

for x in range(len(edges.tolist())):
	for y in range(len(edges.tolist()[0])):
		Matrix[x][y]=1-Matrix[x][y]
		if Matrix[x][y] ==1:
			Matrix[x][y]=1000

dist1,parent1=shortest(Matrix, ROW, COL,max(point1y-ymin-1,0),max(point1x-xmin-1,0))

dist2,parent2=shortest(Matrix, ROW, COL,max(point2y-ymin-1,0),max(point2x-xmin-1,0))

dist3,parent3=shortest(Matrix, ROW, COL,max(point3y-ymin-1,0),max(point3x-xmin-1,0))

dist4,parent4=shortest(Matrix, ROW, COL,max(point4y-ymin-1,0),max(point4x-xmin-1,0))

dist5,parent5=shortest(Matrix, ROW, COL,max(point5y-ymin-1,0),max(point5x-xmin-1,0))


img=printPath(parent1,max(point2y-ymin-1,0),max(point2x-xmin-1,0),img)
img=printPath(parent2,max(point3y-ymin-1,0),max(point3x-xmin-1,0),img)
img=printPath(parent3,max(point4y-ymin-1,0),max(point4x-xmin-1,0),img)
img=printPath(parent4,max(point5y-ymin-1,0),max(point5x-xmin-1,0),img)
printPath(parent1,max(point5y-ymin-1,0),max(point5x-xmin-1,0),img)

cv2.imwrite('left.jpg',img)
print('done')

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
