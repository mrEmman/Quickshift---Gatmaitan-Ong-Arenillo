# SOURCE (entire class) https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
# import the necessary packages
from scipy.spatial import distance as dist
#from sklearn.cluster import KMeans
from collections import OrderedDict
import numpy as np
import cv2

class CentroidTracker():	

	def __init__(self, maxDisappeared=5):
		# initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
		self.color = []
		self.merged_array = []
		self.pastColor = []

		# store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
		self.maxDisappeared = maxDisappeared

	def register(self, centroid):
		# when registering an object we use the next available object
		# ID to store the centroid
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def deregister(self, objectID):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		del self.objects[objectID]
		del self.disappeared[objectID]

	def update(self, rects, color_pixel):
		# check to see if the list of input bounding box rectangles
		# is empty
		if len(rects) == 0:
			# loop over any existing tracked objects and mark them
			# as disappeared
			for objectID in self.disappeared.keys():
				self.disappeared[objectID] += 1

				# if we have reached a maximum number of consecutive
				# frames where a given object has been marked as
				# missing, deregister it
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			# return early as there are no centroids or tracking info
			# to update
			return self.objects

		# initialize an array of input centroids for the current frame
		inputCentroids = np.zeros((len(rects), 2), dtype ="int")
		object_size = np.zeros((len(rects), 2), dtype= "float")
		# loop over the bounding box rectangles
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			# use the bounding box coordinates to derive the centroid
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)
			object_size[i] = ((startX-endX),(startY-endY))

		# if we are currently not tracking any objects take the input
		# centroids and register each of them
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])
				self.color.append(color_pixel[i])
		# otherwise, are are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids
		else:
			# grab the set of object IDs and corresponding centroids
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			# compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid
			D = dist.cdist(np.array(objectCentroids), inputCentroids)
			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value as at the *front* of the index
			# list
			rows = D.min(axis=1).argsort()
			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
			cols = D.argmin(axis=1)[rows]

			# in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
			usedRows = set()
			usedCols = set()

			# loop over the combination of the (row, column) index
			# tuples
			for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
				if row in usedRows or col in usedCols:
					continue

				# otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0

				# indicate that we have examined each of the row and
				# column indexes, respectively
				usedRows.add(row)
				usedCols.add(col)

			# compute both the row and column index we have NOT yet
			# examined
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
			if D.shape[0] > D.shape[1]: #(converge or exit(1270)):
				# loop over the unused row indexes
				for row in unusedRows:
					objectID = objectIDs[row]
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
					d = []
					if self.objects[objectID][0]< 1280 and self.objects[objectID][1]>170:# if the last point is NOT near the border
						print("merge")
						self.disappeared[objectID] += 1
						# check to see if the number of consecutive
						# frames the object has been marked "disappeared"
						# for warrants deregistering the object
						if self.disappeared[objectID] == self.maxDisappeared:
							self.deregister(objectID)
					else: # point is near the border
							self.deregister(objectID)

			# otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
			else:
				for col in unusedCols:
					if inputCentroids[col][1]> 150 and inputCentroids[col][0]>470:# if the point didn't just enter
						print("unmerge")
						# initialize the index dictionary to store the image

						#get image to match, compare with images from previous frame
						index = {}
						hist = cv2.calcHist(color_pixel[col], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
						hist = cv2.normalize(hist,hist).flatten()
						current_ROI = hist #unknown obj fron current frame
						print("rows")
						print(rows)
						break
						for i in rows:
							# extract a 3D RGB color histogram from the image,
							# using 8 bins per channel, normalize, and update
							# the index
							if i in self.disappeared:#not yet disappeared
								print(i)
								hist = cv2.calcHist(self.pastColor[i], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])#calculate color histogram from past objects
								hist = cv2.normalize(hist,hist).flatten()
								index[i]=hist
						results = {}
						# loop over the index

						if len(index)>0:
							for (k, hist) in index.items():
							# compute the distance between the two histograms
								d = cv2.compareHist(current_ROI, hist, cv2.HISTCMP_INTERSECT)
								results[k] = d
							# sort the results
							results = sorted([(v, k) for (k, v) in results.items()], reverse = True)
							print("results")
							print(results[0][1])
							self.objects[results[0][1]]=inputCentroids[col]#record [position]
							self.color.append(color_pixel[col])#record color
							self.disappeared[results[0][1]] = 0#reset disappeared
					else:
						print("entry")
						self.register(inputCentroids[col])
						self.color.append(color_pixel[col])
		self.pastColor = self.color
		# return the set of trackable objects
		return self.objects