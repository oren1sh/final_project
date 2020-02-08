import cv2
import numpy as np
import decimal
import PIL.Image
import csv
import pickle
from decimal import Decimal
from skimage import io
from sklearn.externals import joblib
import DetectorModel
import datetime
import imutils
import os
import math
from scipy.io import loadmat

class GeneratorOfData:


	def __init__(self):
		mat = loadmat('models\\model3D.mat')
		model = mat['model3D']
		self.camera_matrix = np.asmatrix(model['outA'][0, 0], dtype='float32') #3x3
		self.model_TD = np.asarray(model['threedee'][0,0], dtype='float32') #68x3
		self.vectors_of_features = []
		self.vectors = []
		self.fails = []
		self.i = 0


	def compute_features_vectors(self,face_points):


		if numberOfPoints == 68:
			points_of_ancor = np.array([face_points[36],# Right eye right corne
							face_points[39],#Right eye left corne
							face_points[42],# left eye right corne
							face_points[45],# left eye left corne
							face_points[27],# Nose top
							face_points[33],# Nose tip
							face_points[48],# Mouth right corne
							face_points[57],# Mouth botton tip
							face_points[54],# Mouth left corne
							face_points[0],# face up right corne
							face_points[8],# face botton corne
							face_points[16]],# face up left corne
								dtype="double")
			points_of_ancor_vectors = np.zeros((12,12,3))
			for i in range(12):
				for j in range (12):
					vectors_parts = [0.,0.,0.]
					if i == j:
						points_of_ancor_vectors[i][j][:] = vectors_parts
						continue
					distance = [points_of_ancor[i][0] - points_of_ancor[j][0], points_of_ancor[i][1] - points_of_ancor[j][1]]
					norm = math.sqrt(distance[0] ** 2 + distance[1] ** 2)#size of vec
					vectors_parts[0] = norm
					vectors_parts[1] = distance[0] / norm
					vectors_parts[2] = distance[1] / norm

					points_of_ancor_vectors[i][j][:] = vectors_parts
				
			
			return np.array(points_of_ancor_vectors).reshape(1, -1)



	def solveThePNP(self,image,marks):

		size = image.shape
		#2D image points.  If you change the image, you need to change vector
		image_points = np.array([marks[33],     # Nose tip
								marks[8],     # Chin
								marks[36],     # Left eye left corner
								marks[45],     # Right eye right corne
								marks[48],     # Left Mouth corner
								marks[54]      # Right mouth corner
							], dtype="double")
		image_points = np.array(
						   [marks[36],# Right eye right corne
							marks[39],#Right eye left corne
							marks[42],# left eye right corne
							marks[45],# left eye left corne
							marks[27],# Nose top
							marks[33],# Nose tip
							marks[48],# Mouth right corne
							marks[57],# Mouth botton tip
							marks[54],# Mouth left corne
							marks[0],# face up right corne
							marks[8],# face botton corne
							marks[16]],# face up left corne
								dtype="double")


	
		# 3D model points.
		model_points = np.array([self.model_TD[33],    # Nose tip
								self.model_TD[8],      # Chin
								self.model_TD[36],     # Left eye left corner
								self.model_TD[45],     # Right eye right corne
								self.model_TD[48],     # Left Mouth corner
								self.model_TD[54]      # Right mouth corner
							], dtype="double")
		model_points = np.array(
						   [self.model_TD[36],# Right eye right corne
							self.model_TD[39],#Right eye left corne
							self.model_TD[42],# left eye right corne
							self.model_TD[45],# left eye left corne
							self.model_TD[27],# Nose top
							self.model_TD[33],# Nose tip
							self.model_TD[48],# Mouth right corne
							self.model_TD[57],# Mouth botton tip
							self.model_TD[54],# Mouth left corne
							self.model_TD[0],# face up right corne
							self.model_TD[8],# face botton corne
							self.model_TD[16]],# face up left corne
								dtype="double")

		dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
			
		(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, self.camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
		return[rotation_vector,translation_vector]


	def getMarks(self,image,face_locations):

		face_landmarks_list = DetectorModel.face_landmarks(image,face_locations)[0]
		marks = []
	
		#set the points in marks
		for facial_feature in face_landmarks_list.keys():
			for element in face_landmarks_list[facial_feature]:
				marks.append(element)
		return marks

	def getFaceLocations(self,image,location):
		print("get face locations " + str(datetime.datetime.now()))
		face_locations = DetectorModel.face_locations(image,number_of_times_to_upsample = 0, model="cnn")
		if len(face_locations) == 0 :
			print("FAIL!")
			self.fails.append(location)
			return None
		return face_locations


	def run(self):
		for dirpath, dirnames, filenames in os.walk("ImageData\\CropDataImage\\"):
			for filename1 in [f for f in filenames if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")]:
				location = dirpath + "\\" + filename1
				currentTime = datetime.datetime.now()
				try:
					image = DetectorModel.load_image_file(location)
					print("############################### " + str(self.i) + " ##############################")
					print(location + " " + str(datetime.datetime.now()) + ":")
					face_locations = self.getFaceLocations(image,location)
					if(face_locations == None):
						self.i = self.i + 1
						continue
					#get the points
					marks = self.getMarks(image,face_locations)

					#compute and add the features to the array
					self.vectors_of_features.append(self.compute_features_vectors(marks)[0])

					pnpvectors = self.solveThePNP(image,marks)
					rVector = pnpvectors[0]
					tVector = pnpvectors[1]
					#add the vectors
					self.vectors.append([rVector[0][0],
									rVector[1][0],
									rVector[2][0],
									tVector[0][0],
									tVector[1][0],
									tVector[2][0]])	
					print(" ################### SUCCESS " + str(datetime.datetime.now()) + "#######################")
					if self.i % 200 == 0:
						print(str(datetime.datetime.now()))
						print(" ################### Saving Point " + str(datetime.datetime.now()) + "#######################")
						print("i = " + str(self.i))
						outfile = open('OutputsData\\NetModeTest.pkl','wb')
						pickle.dump((self.vectors_of_features,rVector,tVector),outfile)
						outfile.close()
						print("self.vectors_of_features = " + str(len(self.vectors_of_features)))
						print("self.fails = " + str(self.fails))

					self.i = self.i + 1
				except Exception as e:
					print(str(e))
					self.i = self.i + 1
					self.fails.append(location)
	
			outfile = open('OutputsData\\NetModeTest.pkl','wb')
			pickle.dump((self.vectors_of_features,rVector,tVector),outfile)
			outfile.close()
			print("vectors_of_features = " + str(len(self.vectors_of_features)))
			print("fails = " + str(self.fails))



