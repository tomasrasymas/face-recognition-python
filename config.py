import os

image_path = 'images'
original_images_path = os.path.join(image_path, 'original')
training_images_path = os.path.join(image_path, 'training')
models_path = 'models'
dlib_models_path = os.path.join(models_path, 'dlib')
dlib_shape_predictor_model = os.path.join(dlib_models_path, 'shape_predictor_68_face_landmarks.dat')
number_of_faces = 10
face_size = (300, 300)
trained_model = os.path.join(models_path, 'trained_model.pkl')