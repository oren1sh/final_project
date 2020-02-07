from Model3DHelper import Model3DHelper
import datetime
import numpy as np
import os
import FacesHelper
import cv2 as cv2
class SolvePNPManager:
	def __init__(self):	
		self.model3D = Model3DHelper()
	def getImagePoints(self, marks):
		"""
		Returns np array with the relevant points (from 68 or 5)
		"""
		if len(marks) == 5: #get the 5 points
			return np.array([marks[4],   # Nose tip
							marks[2],     # Left eye left corner
							marks[3],     # Left eye right corner
							marks[0],     # Right eye left corner
							marks[1],     # Right eye right corner
				], dtype="double")
		#by default, get the 68 points
		return np.array([marks[33],    # Nose tip
						marks[8],      # Chin
						marks[36],     # Left eye left corner
						marks[45],     # Right eye right corner
						marks[48],     # Left Mouth corner
						marks[54]      # Right mouth corner
						], dtype="double")
	def getMarks(self,image,face_locations):
		"""
		Returns the marks of the image
		"""
		marks = []
		try:
			face_landmarks_list = FacesHelper.face_landmarks(image,face_locations,model = 'large')[0]
			for facial_feature in face_landmarks_list.keys():
				for element in face_landmarks_list[facial_feature]:
					marks.append(element)
		except: #if can't find 68 points, try to get 5 points
			face_landmarks_list = FacesHelper.face_landmarks(image,face_locations,model = 'small')[0]
			for facial_feature in face_landmarks_list.keys():
				for element in face_landmarks_list[facial_feature]:
					marks.append(element)
		return marks

	def predict(self):
		fails = []
		from CSVHelper import CSVHelper
		csvHelper = CSVHelper("SOLVE_PNP")
		i = 0 
		print("################### START PREDICT SOLVE PNP " + str(datetime.datetime.now()) + " #####################")
		for dirpath, dirnames, filenames in os.walk("InputData\\"):
			for filename1 in [f for f in filenames if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")]:
				filename = dirpath + "\\" + filename1
				newFileName = filename.replace("InputData\\","")
				try:
					image = FacesHelper.load_image_file(filename)
					#get the face locations, try with 68 points, then with 5 points
					face_locations = FacesHelper.face_locations(image,number_of_times_to_upsample = 0, model="cnn")
					if len(face_locations) == 0:#try with the HOG
						face_locations = FacesHelper.face_locations(image,number_of_times_to_upsample = 0, model="HOG")
					#no face was found
					if len(face_locations) == 0 :
						print(newFileName + " - NO FACES FOUND!! \n " + str(ex))
						fails.append(newFileName)
						continue
					#end if no face was found
					marks = self.getMarks(image,face_locations)#get the marks for the image

					#if no marks, failed with this image
					if len(marks) == 0:
						print(newFileName + " - NO FACES FOUND!! \n " + str(ex))
						fails.append(newFileName)
						continue

					#solve with PNP:
					imagePoints = marks[0:68]
					#solve PNP with 68 or 5 points according to the length of the marks
					(success, rotation_vector, translation_vector) = cv2.solvePnP(self.model3D.model3DPoints5 if len(marks) == 5 else self.model3D.model3DPoints68, 
						imagePoints, 
						self.model3D.camera_matrix, 
						self.model3D.dist_coeffs, 
						flags=cv2.SOLVEPNP_ITERATIVE)
					#adding the row to the csv
					csvHelper.addRow(i,	
					  newFileName, 
					  rotation_vector[0][0], 
					  rotation_vector[1][0],
					  rotation_vector[2][0],
					  translation_vector[0][0],
					  translation_vector[1][0],
					  translation_vector[2][0])
					i = i + 1
				except Exception as ex:
					print(filename + " - ERROR ACCURED!! \n " + str(ex))
					fails.append(newFileName)
					continue
		csvHelper.writeToCSV()
		print()
		print(" SUMMARY : ")
		print("TOTAL SUCCESS = " + str(len(csvHelper.rows) - 1))
		print("TOTAL FAILS = " + str(len(fails)))
		print()
		print("################### END PREDICT SOLVE PNP  " + str(datetime.datetime.now()) + " #####################")

