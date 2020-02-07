import PIL.Image
import dlib
import numpy as np
import math
face_detector = dlib.get_frontal_face_detector()

pose_predictor_68_point = dlib.shape_predictor("Models\\shape_predictor_68_face_landmarks.dat")
pose_predictor_5_point = dlib.shape_predictor("Models\\shape_predictor_5_face_landmarks.dat")
cnn_face_detector = dlib.cnn_face_detection_model_v1("Models\\mmod_human_face_detector.dat")
	
def _rect_to_css(rect):
	"""
	Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order

	:param rect: a dlib 'rect' object
	:return: a plain tuple representation of the rect in (top, right, bottom, left) order
	"""
	return rect.top(), rect.right(), rect.bottom(), rect.left()

def _css_to_rect(css):
	"""
	Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object

	:param css:  plain tuple representation of the rect in (top, right, bottom, left) order
	:return: a dlib `rect` object
	"""
	return dlib.rectangle(css[3], css[0], css[1], css[2])

def _trim_css_to_bounds(css, image_shape):
	"""
	Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.

	:param css:  plain tuple representation of the rect in (top, right, bottom, left) order
	:param image_shape: numpy shape of the image array
	:return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
	"""
	return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)

def load_image_file(file, mode='RGB'):
	"""
	Loads an image file (.jpg, .png, etc) into a numpy array

	:param file: image file name or file object to load
	:param mode: format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L' (black and white) are supported.
	:return: image contents as numpy array
	"""
	im = PIL.Image.open(file)
	if mode:
		im = im.convert(mode)
	return np.array(im)

def _raw_face_locations(img, number_of_times_to_upsample=1, model="hog"):
	"""
	Returns an array of bounding boxes of human faces in a image

	:param img: An image (as a numpy array)
	:param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
	:param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
					deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
	:return: A list of dlib 'rect' objects of found face locations
	"""
	if model == "cnn":
		return cnn_face_detector(img, number_of_times_to_upsample)
	else:
		return face_detector(img, number_of_times_to_upsample)

def face_locations(img, number_of_times_to_upsample=0, model="hog"):
	"""
	Returns an array of bounding boxes of human faces in a image

	:param img: An image (as a numpy array)
	:param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
	:param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
					deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
	:return: A list of tuples of found face locations in css (top, right, bottom, left) order
	"""
	if model == "cnn":
		return [_trim_css_to_bounds(_rect_to_css(face.rect), img.shape) for face in _raw_face_locations(img, number_of_times_to_upsample, "cnn")]
	else:
		return [_trim_css_to_bounds(_rect_to_css(face), img.shape) for face in _raw_face_locations(img, number_of_times_to_upsample, model)]

def _raw_face_locations_batched(images, number_of_times_to_upsample=1, batch_size=128):
	"""
	Returns an 2d array of dlib rects of human faces in a image using the cnn face detector

	:param img: A list of images (each as a numpy array)
	:param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
	:return: A list of dlib 'rect' objects of found face locations
	"""
	return cnn_face_detector(images, number_of_times_to_upsample, batch_size=batch_size)

def batch_face_locations(images, number_of_times_to_upsample=1, batch_size=128):
	"""
	Returns an 2d array of bounding boxes of human faces in a image using the cnn face detector
	If you are using a GPU, this can give you much faster results since the GPU
	can process batches of images at once. If you aren't using a GPU, you don't need this function.

	:param img: A list of images (each as a numpy array)
	:param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
	:param batch_size: How many images to include in each GPU processing batch.
	:return: A list of tuples of found face locations in css (top, right, bottom, left) order
	"""
	def convert_cnn_detections_to_css(detections):
		return [_trim_css_to_bounds(_rect_to_css(face.rect), images[0].shape) for face in detections]

	raw_detections_batched = _raw_face_locations_batched(images, number_of_times_to_upsample, batch_size)

	return list(map(convert_cnn_detections_to_css, raw_detections_batched))

def _raw_face_landmarks(face_image, face_locations=None, model="large"):
	if face_locations is None:
		face_locations = _raw_face_locations(face_image)
	else:
		face_locations = [_css_to_rect(face_location) for face_location in face_locations]

	pose_predictor = pose_predictor_68_point

	if model == "small":
		pose_predictor = pose_predictor_5_point

	return [pose_predictor(face_image, face_location) for face_location in face_locations]

def face_landmarks(face_image, face_locations=None, model="large"):
	"""
	Given an image, returns a dict of face feature locations (eyes, nose, etc) for each face in the image

	:param face_image: image to search
	:param face_locations: Optionally provide a list of face locations to check.
	:param model: Optional - which model to use. "large" (default) or "small" which only returns 5 points but is faster.
	:return: A list of dicts of face feature locations (eyes, nose, etc)
	"""
	landmarks = _raw_face_landmarks(face_image, face_locations, model)
	landmarks_as_tuples = [[(p.x, p.y) for p in landmark.parts()] for landmark in landmarks]

	# For a definition of each point index, see
	# https://cdn-images-1.medium.com/max/1600/1*AbEg31EgkbXSQehuNJBlWg.png
	if model == 'large':
		return [{
			"chin": points[0:17],
			"left_eyebrow": points[17:22],
			"right_eyebrow": points[22:27],
			"nose_bridge": points[27:31],
			"nose_tip": points[31:36],
			"left_eye": points[36:42],
			"right_eye": points[42:48],
			"top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
			"bottom_lip": points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] + [points[64]]
		} for points in landmarks_as_tuples]
	elif model == 'small':
		return [{
			"nose_tip": [points[4]],
			"left_eye": points[2:4],
			"right_eye": points[0:2],
		} for points in landmarks_as_tuples]
	else:
		raise ValueError("Invalid landmarks model type. Supported models are ['small', 'large'].")

def compute_features(face_points,expectedNumberOfPoints=68):
	"""
	Computing euclidian distances between all points
	"""
	assert (len(face_points) >= expectedNumberOfPoints), "len(face_points) must be at least " + str(expectedNumberOfPoints)
	
	face_points = np.array(face_points)
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
					face_points[16]], dtype="double")
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
				
	print( "Good run")
	return np.array(points_of_ancor_vectors).reshape(1, -1)
			


