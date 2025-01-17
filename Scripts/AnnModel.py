


import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, ion, show
import _pickle as pkl
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import DetectorModel
from CSVModel import CSVModel
import datetime
import os
class AnnModel:
	def __init__(self):	
		data = pkl.load(open('OutputsData\\NetModel300.pkl', 'rb'))
		self.x = np.array(data[0])#each image has array distence vectors
		self.y_r_v = np.array(data[1])#each image has the rotation vector
		self.y_t_v = np.array(data[2])#each image has the translation vector

		self.rot_model = load_model('Models\\RotTrainedNetModel.h5')
		self.trans_model = load_model('Models\\TransTrainedNetModel.h5')

	def print_shapes(self,x_train,y_train,x_val,y_val,x_test,y_test):
		"""
		Printing the shapes of the sets
		"""
		print(x_train.shape, y_train.shape)
		print(x_val.shape, y_val.shape)
		print(x_test.shape, y_test.shape)

	def showTrainingGraph(self,hist):
		history = hist.history
		loss_train = history['loss']
		loss_val = history['val_loss']
		
		plt.figure()
		ion() # enables interactive mode
		plot(loss_train, label='train')
		plot(loss_val, label='val_loss', color='red')
		plt.legend()
		show()
		return loss_train, loss_val

	def showRotationVectorRotGraph(self, diff_pitch, diff_roll, diff_yaw, loss_train, loss_val):# , diff_ty ,diff_tx ,diff_tr):
		plt.figure(figsize=(16, 10))														 
		plt.plot(loss_train, label='trainOfRot')													 
		plt.plot(loss_val, label='val_loss', color='red')
		plt.legend()
		plt.subplot(3, 1, 1)
		plt.plot(diff_roll, color='red')
		plt.title('roll')
		
		plt.subplot(3, 1, 2)
		plt.plot(diff_pitch, color='red')
		plt.title('pitch')
		
		plt.subplot(3, 1, 3)
		plt.plot(diff_yaw, color='red')
		plt.title('yaw')
		
		plt.tight_layout()
		plt.show()
		input("wait")



	def showRotationVectorTranGraph(self, diff_ty, diff_tx, diff_tr, loss_train, loss_val):# , diff_ty ,diff_tx ,diff_tr):
		plt.figure(figsize=(16, 10))														 
		plt.plot(loss_train, label='trainOfTrans')													 
		plt.plot(loss_val, label='val_loss', color='red')
		plt.legend()
		
		plt.subplot(3, 2, 1)
		plt.plot(diff_ty, color='red')
		plt.title('ty')
		
		plt.subplot(3, 2, 2)
		plt.plot(diff_tx, color='red')
		plt.title('tx')
		
		plt.subplot(3, 2, 3)
		plt.plot(diff_tr, color='red')
		plt.title('tr')
		
		plt.tight_layout()
		plt.show()
		input("wait")


	def train(self,BATCH_SIZE=32, EPOCHS=5000):
		print("###################  TRAINING START AT ==> " + str(datetime.datetime.now()) + " #####################")
		


		#divide you dataset in train, validation, and test
		x_train, x_test, y_train, y_test = train_test_split(self.x, self.y_r_v, test_size=0.2, random_state=42)
		x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.2, random_state=42)

		# print the shapes of the sets
		self.print_shapes(x_train,y_train,x_val,y_val,x_test,y_test) 

		self.setGlobalSTD(x_train)
		x_val = self.std.transform(x_val)
		x_test = self.std.transform(x_test)

		#Training the by rotation vector
		model = Sequential()
		input_dim = self.x.shape[1] 
		model.add(Dense(units=432, activation='softsign', input_dim=input_dim))
		model.add(Dense(units=432, activation='linear'))
		model.add(Dense(units=60, activation='linear'))
		model.add(Dense(units=36, activation='linear'))
		model.add(Dense(units=3, activation='linear'))

		print(model.summary())

		#print status bar of the progress

		callback_list = [EarlyStopping(monitor='val_loss', patience=50)]
		model.compile(optimizer = 'adam', loss='mean_squared_error',metrics=['mae', 'acc'])
		#training with the fit function
		hist = model.fit(x=self.x_train, y=y_train, validation_data=(x_val, y_val), batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callback_list,use_multiprocessing=True)
		model.save('Models\\RotTrainedNetModel.h5')#saving the rotation vector model

		print()
		print('Train loss:', model.evaluate(self.x_train, y_train, verbose=0))
		print('Val loss:', model.evaluate(x_val, y_val, verbose=0))
		print('Test loss:', model.evaluate(x_test, y_test, verbose=0))

		#visualize training graph:
		loss_train, loss_val = self.showTrainingGraph(hist)

		#plot the difference between expected and predictions values for rotation vector:
		y_pred = model.predict(x_test)#get predict of the rotation vector
		diff = y_test - y_pred
		diff_roll = diff[:, 0]
		diff_pitch = diff[:, 1]
		diff_yaw = diff[:, 2]
		self.showRotationVectorRotGraph(diff_pitch, diff_roll, diff_yaw,loss_train,loss_val)#


		#divide you dataset in train, validation, and test
		x_train, x_test, y_train, y_test = train_test_split(self.x, self.y_t_v, test_size=0.2, random_state=42)
		x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.2, random_state=42)

		self.print_shapes(x_train,y_train,x_val,y_val,x_test,y_test) # print the shapes of the sets

		self.setGlobalSTD(x_train)
		x_val = self.std.transform(x_val)
		x_test = self.std.transform(x_test)

		#Training
		model = Sequential()
		input_dim = self.x.shape[1] #train by the x dimension
		model.add(Dense(units=432, activation='softsign', input_dim=input_dim))
		model.add(Dense(units=432, activation='linear'))
		model.add(Dense(units=432, activation='linear'))
		model.add(Dense(units=432, activation='linear'))
		model.add(Dense(units=100, activation='linear'))
		model.add(Dense(units=100, activation='linear'))
		model.add(Dense(units=100, activation='linear'))
		model.add(Dense(units=60, activation='linear'))
		model.add(Dense(units=36, activation='linear'))
		model.add(Dense(units=3, activation='linear'))

		print(model.summary())

		#print status bar of the progress

		callback_list = [EarlyStopping(monitor='val_loss', patience=50)]
		model.compile(optimizer = 'adam', loss='mean_squared_error',metrics=['mae', 'acc'])
		#training with the fit function
		hist = model.fit(x=self.x_train, y=y_train, validation_data=(x_val, y_val), batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callback_list,use_multiprocessing=True)
		model.save('Models\\TransTrainedNetModel.h5')#saving the model

		print()
		print('Train loss:', model.evaluate(self.x_train, y_train, verbose=0))
		print('Val loss:', model.evaluate(x_val, y_val, verbose=0))
		print('Test loss:', model.evaluate(x_test, y_test, verbose=0))

		#visualize training graph:
		loss_train, loss_val = self.showTrainingGraph(hist)

		#plot the difference between expected and predictions values:
		y_pred = model.predict(x_test)#get predict of the rotation and translation vector
		diff = y_test - y_pred
		diff_ty = diff[:, 0]
		diff_tx = diff[:, 1]
		diff_tr = diff[:, 2]
		self.showRotationVectorTranGraph(diff_ty, diff_tx, diff_tr,loss_train,loss_val)#,diff_ty,diff_tx,diff_tr)

		print("################### END OF TRAINING AT TIME ==> " + str(datetime.datetime.now()) + " #####################")

	def setGlobalSTD(self, x_train):
		"""
		setting fitted standard scalar
		"""
		self.std = StandardScaler()
		self.x_train = x_train
		self.std.fit(x_train)
		self.x_train = self.std.transform(self.x_train)

	def predictByModel(self):
		"""
		predicting the output by the trained model
		"""
		CSVModel = CSVModel("ANN")
		print("################### START PREDICT BY MODEL " + str(datetime.datetime.now()) + " #####################")
		self.rot_model = load_model('Models\\RotTrainedNetModel.h5')
		self.trans_model = load_model('Models\\TransTrainedNetModel.h5')
		fails = []
		i = 0
		self.setGlobalSTD(self.x)
		for dirpath, dirnames, filenames in os.walk("InputData"):
			for filename1 in [f for f in filenames if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")]:
				filename = dirpath + "\\" + filename1
				newFileName = filename.replace("InputDatav\\","")#set the new file name for the CSV
				image = DetectorModel.load_image_file(filename)#load the image
				face_locations = DetectorModel.face_locations(image,number_of_times_to_upsample = 0, model="cnn")#detect face locations using CNN
				landmarks = DetectorModel.face_landmarks(image,face_locations)#detect the landmarks of the face
				face_points = []
				for key in landmarks[0]:
					for point in landmarks[0][key]:
						face_points.append(point)
				if len(face_points) < 68:#if face is not detected, continue
					print('\n\n ***************  ' + filename + ' can NOT be recognized as a full face ***************\n')
					fails.append(newFileName)
					continue
				features = DetectorModel.compute_features(face_points = face_points)#compute euclidian distances between the points
				features = self.std.transform(features)

				y_pred = model.rot_model(features,use_multiprocessing=True)#predict the values by the trained model
				rx,ry,rz = y_pred[0]

				y_pred = model.trans_model(features,use_multiprocessing=True)#predict the values by the trained model
				tx,ty,tz = y_pred[0]

				CSVModel.addRow(i,newFileName,rx,ry,rz,tx,ty,tz)
				i = i + 1
		CSVModel.writeToCSV()
		print("################### END PREDICT AT TIME ==>" + str(datetime.datetime.now()) + " #####################")
