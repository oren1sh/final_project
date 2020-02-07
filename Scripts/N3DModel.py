from scipy.io import loadmat
import numpy as np
class N3DModel:
	def __init__(self):	
		mat = loadmat('Models\\model3D.mat')#loading the model
		model = mat['model3D']
		self.camera_matrix = np.asmatrix(model['outA'][0, 0], dtype='float32') #3x3 - the camera matrix of the model
		self.model3D = np.asarray(model['threedee'][0,0], dtype='float32') #68x3 - the face landmarks of the 3D model
		self.model3DPoints68 = self.model3D #set the relevant points from the 3D model
		self.model3DPoints5 = np.array([self.model3D[33],   # Nose tip
								  	self.model3D[36],     # Left eye left corner
								  	self.model3D[39],     # Left eye right corner
									self.model3D[42],     # Right eye left corner
									self.model3D[45],     # Right eye right corner
									], dtype="double") #set the relevant 5 points from the 3D model

		self.dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
