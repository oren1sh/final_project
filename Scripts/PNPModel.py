

from N3DModel import N3DModel
import datetime
import numpy as np
import os
import DetectorModel
#from DetectorModel import DetectorModel
import cv2 as cv2
from CSVModel import CSVModel
import time

class PNPModel:
	def __init__(self):	
		self.model3D = N3DModel()
		#self.DetectorModel = DetectorModel()
		self.CSVModel = CSVModel("SOLVE_PNP")


	def getImagePoints(self, marks):

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
		return image_points



	def getMarks(self,image,face_locations):
		"""
		Returns the marks of the image
		"""
		marks = []
		try:
			face_landmarks_list = DetectorModel.face_landmarks(image,face_locations,model = 'large')[0]
			for facial_feature in face_landmarks_list.keys():
				for element in face_landmarks_list[facial_feature]:
					marks.append(element)
		except: #if can't find 68 points, try to get 5 points
			face_landmarks_list = DetectorModel.face_landmarks(image,face_locations,model = 'small')[0]
			for facial_feature in face_landmarks_list.keys():
				for element in face_landmarks_list[facial_feature]:
					marks.append(element)
		return marks

	def predict(self):
		fails = []
		start_time = time.clock()
		end_time = time.clock()
		i = 0 
		for dirpath, dirnames, filenames in os.walk("InputData\\test_set\\"):
			for filename1 in [f for f in filenames if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")]:
				filename = dirpath + "\\" + filename1
				newFileName = filename.replace("InputData\\test_set\\","")
				start_time = time.clock()
				print("################### START PNP ON " + newFileName +  " AT TIME " + str(datetime.datetime.now()) + " #####################" + "\n")
				try:
					image = DetectorModel.load_image_file(filename)
					#get the face locations, try with 68 points, then with 5 points
					face_locations = DetectorModel.face_locations(image,number_of_times_to_upsample = 0, model="cnn")
					if len(face_locations) == 0:#try with the HOG
						face_locations = DetectorModel.face_locations(image,number_of_times_to_upsample = 0, model="HOG")
					#no face was found
					if len(face_locations) == 0 :
						print(newFileName + " - NO FACES FOUND!!! \n " + str(ex))
						fails.append(newFileName)
						continue
					#end if no face was found
					marks = self.getMarks(image,face_locations)#get the marks for the image

					#if no marks, failed with this image
					if len(marks) == 0:
						print(newFileName + " - NO FACES FOUND!!! \n " + str(ex))
						fails.append(newFileName)
						continue

					#solve with PNP:
					imagePoints = np.float32(marks[0:68])

					#solve PNP with 68 points according to the length of the marks
					(success, rotation_vector, translation_vector) = cv2.solvePnP(self.model3D.model3DPoints5 if len(marks) == 5 else self.model3D.model3DPoints68, 
						imagePoints, 
						self.model3D.camera_matrix, 
						self.model3D.dist_coeffs, 
						flags=cv2.SOLVEPNP_ITERATIVE)
					#adding the row to the csv
					self.CSVModel.addRow(i,	
					  newFileName, 
					  rotation_vector[0][0], 
					  rotation_vector[1][0],
					  rotation_vector[2][0],
					  translation_vector[0][0],
					  translation_vector[1][0],
					  translation_vector[2][0])
					i = i + 1
					end_time = time.clock() - start_time
					print("################### PNP SUCCESS ON " + newFileName +  " TOTAL TIME TAKE " + str(end_time) + " #####################"+"\n")
				except Exception as ex:
					end_time = time.clock() - start_time
					print(filename + " - ERROR !!! \n " + str(ex))
					print("################### PNP FAIL ON " + newFileName +  " TOTAL TIME TAKE " + str(end_time) + " #####################" +"\n")
					fails.append(newFileName)
					continue
		self.CSVModel.writeToCSV()
		print()
		print(" SUMMARY : ")
		print("TOTAL SUCCESS = " + str(len(self.CSVModel.rows) - 1))
		print("TOTAL FAILS = " + str(len(fails)))
		print()
		print("################### END PNP  " + str(datetime.datetime.now()) + " #####################")


