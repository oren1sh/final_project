import cv2
import numpy as np
import decimal
import PIL.Image
import csv
import pickle
from decimal import Decimal
from skimage import io
from sklearn.externals import joblib
import FacesHelper
import datetime
import imutils
import os
import math
from scipy.io import loadmat

class DataGeneratorModel:


	def __init__(self,methodName):

		mat = loadmat('models\\model3D.mat')
		model = mat['model3D']
		camera_matrix = np.asmatrix(model['outA'][0, 0], dtype='float32') #3x3
		model_TD = np.asarray(model['threedee'][0,0], dtype='float32') #68x3
		globalFeatures = []
		fiveFeatures = []
		croppedFeatures = []
		globalFeaturesCounter = 0
		vectors = []
		fails = []
		i = 0
		notcropped = []

	def compute_features(face_points,numberOfPoints=68):


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
					#direction = [distance[0] / norm, distance[1] / norm]
					#vectors_parts.append(direction)
					#print( "vectors_parts ==  " + str(vectors_parts))
					points_of_ancor_vectors[i][j][:] = vectors_parts
				
			#print( "points_of_ancor_vectors ==  " + str(points_of_ancor_vectors))
			return np.array(points_of_ancor_vectors).reshape(1, -1)


		if numberOfPoints == 5:
				face_points = np.array([face_points[33],   # Nose tip
								  		face_points[36],     # Left eye left corner
								  		face_points[39],     # Left eye right corner
										face_points[42],     # Right eye left corner
										face_points[45]]) #right eye right corner
				features = []
				for i in range(5):
					for j in range(i + 1, 5):
						features.append(np.linalg.norm(face_points[i] - face_points[j]))
			
				return np.array(features).reshape(1, -1)
		#print("compute features " + str(datetime.datetime.now()))
		assert (len(face_points) >= 68), "len(face_points) must be at least 68"
	
		face_points = np.array(face_points)
		features = []
		for i in range(68):
			for j in range(i + 1, 68):
				features.append(np.linalg.norm(face_points[i] - face_points[j]))
			
		return np.array(features).reshape(1, -1)


	def solveThePNP(image,marks):
		global globalFeatures
		global fiveFeatures
		global croppedFeatures
		global globalFeaturesCounter
		global vectors
		global fails
		global i
		global model_TD
		global camera_matrix
		size = image.shape
		############translation:
		#2D image points.  If you change the image, you need to change vector
		#print("solvePNP " + str(datetime.datetime.now()))
		image_points = np.array([marks[33],     # Nose tip
								marks[8],     # Chin
								marks[36],     # Left eye left corner
								marks[45],     # Right eye right corne
								marks[48],     # Left Mouth corner
								marks[54]      # Right mouth corner
							], dtype="double")
	
		# 3D model points.
		model_points = np.array([model_TD[33],    # Nose tip
								model_TD[8],      # Chin
								model_TD[36],     # Left eye left corner
								model_TD[45],     # Right eye right corne
								model_TD[48],     # Left Mouth corner
								model_TD[54]      # Right mouth corner
							], dtype="double")

		#print("\n\n\n")
		#print("Camera Matrix :\n " + str(camera_matrix) + "\n")
		dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
			
		(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
		return[rotation_vector,translation_vector]


	def getMarks(image,face_locations):
		global globalFeatures
		global croppedFeatures
		global globalFeaturesCounter
		global vectors
		global fails
		global i
		#print("get marks " + str(datetime.datetime.now()))
		face_landmarks_list = FacesHelper.face_landmarks(image,face_locations)[0]
		marks = []
	
		#set the points in marks
		for facial_feature in face_landmarks_list.keys():
			for element in face_landmarks_list[facial_feature]:
				marks.append(element)
		return marks

	def getFaceLocations(image,location):
		global globalFeatures
		global croppedFeatures
		global globalFeaturesCounter
		global vectors
		global fails
		global i
		print("get face locations " + str(datetime.datetime.now()))
		face_locations = FacesHelper.face_locations(image,number_of_times_to_upsample = 0, model="hog")
		if len(face_locations) == 0 :
			print("FAIL!")
			fails.append(location)
			return None
		return face_locations


	def run():
		global globalFeatures
		global croppedFeatures
		global globalFeaturesCounter
		global fiveFeatures
		global vectors
		global fails
		global notcropped
		global i
		for dirpath, dirnames, filenames in os.walk("images\\cropped\\"):
			for filename1 in [f for f in filenames if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")]:
				location = dirpath + "\\" + filename1
				currentTime = datetime.datetime.now()
				try:

					newFileName = location.replace("images\\","")
					image = FacesHelper.load_image_file(location)
					print("############################### " + str(i) + "##############################")
					print(location + " " + str(datetime.datetime.now()) + ":")
					face_locations = getFaceLocations(image,location)
					if(face_locations == None):
						i = i + 1
						continue
					#get the points
					marks = getMarks(image,face_locations)

					#compute and add the features to the array
					#print("############################### " + "start of features finding" + "##############################")
					globalFeatures.append(compute_features(marks)[0])
					#fiveFeatures.append(compute_features(marks,5)[0])
					#print("############################### " + "end of features finding" + "##############################")

					#croppedFeatures.append(compute_features(croppedMarks)[0])
					pnpvectors = solveThePNP(image,marks)
					rVector = pnpvectors[0]
					tVector = pnpvectors[1]
					#add the vectors
					vectors.append([rVector[0][0],
									rVector[1][0],
									rVector[2][0],
									tVector[0][0],
									tVector[1][0],
									tVector[2][0]])	
					print(" ################### SUCCESS " + str(datetime.datetime.now()) + "#######################")
					if i % 200 == 0:
						print(str(datetime.datetime.now()))
						print("i = " + str(i))
						print("globalFeatures = " + str(len(globalFeatures) + globalFeaturesCounter))
						print("vectors = " + str(len(vectors)))
						print("fails = " + str(len(fails)))
						print("not cropped = " + str(len(fails)))

						outfile = open('OutputData\\NetModel300.pkl','wb')
						pickle.dump((globalFeatures,fiveFeatures,vectors),outfile)
						outfile.close()
						print("globalFeatures = " + str(len(globalFeatures) + globalFeaturesCounter))
						print("fails = " + str(fails))
						print("not cropped: " + str(notcropped))

					i = i + 1
					#print(location + " = " + str(im.shape))
				except Exception as e:
					print(str(e))
					i = i + 1
					fails.append(location)
	
		outfile = open('OutputData\\NetModel300.pkl','wb')
		pickle.dump((globalFeatures,fiveFeatures,vectors),outfile)
		outfile.close()
		print("globalFeatures = " + str(len(globalFeatures) + globalFeaturesCounter))
		print("fails = " + str(fails))
		print("not cropped: " + str(notcropped))



