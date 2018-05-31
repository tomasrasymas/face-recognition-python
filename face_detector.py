import numpy as np
import dlib
import cv2
import os
import config
from align_dlib import AlignDlib
from sklearn.externals import joblib
import face_recognition

dlib_face_detector = dlib.get_frontal_face_detector()
face_aligner = AlignDlib(config.dlib_shape_predictor_model)
face_pose_predictor = dlib.shape_predictor(config.dlib_shape_predictor_model)

face_features = face_recognition.FaceFeatures()


def pipeline(image, face, features=False):
    f = cut_facxe(image, face)
    f = to_gray(f)
    f = normalize_intensity(f)
    f = resize(f, config.face_size)
    f = align_face(f)

    if features:
        f = image_to_features(f)

        return [f]

    return f


def image_to_features(image):
    return face_features.describe(image)


def to_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def collect_original_images():
    images = {}

    people = [person for person in os.listdir(config.original_images_path)]
    for person in people:
        images[person] = []

        for image in os.listdir(os.path.join(config.original_images_path, person)):
            images[person].append([image,
                                   cv2.imread(os.path.join(config.original_images_path, person, image), 1)])

    return images


def collect_training_dataset():
    images = []
    labels = []

    people = [person for person in os.listdir(config.training_images_path)]
    for person in people:
        for image in os.listdir(os.path.join(config.training_images_path, person)):
            images.append(cv2.imread(os.path.join(config.training_images_path, person, image), 0))
            labels.append(person)

    return images, np.array(labels)


def face_landmarks_detector(image):
    landmarks = face_pose_predictor(image, image.shape[0])
    return landmarks


def align_face(image):
    aligned_face = face_aligner.align(image.shape[0], image, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
    if aligned_face is None:
        return image
    return aligned_face


def detect_faces_dlib(image):
    detected_faces = dlib_face_detector(image, 1)
    return [(f.left(), f.top(), f.width(), f.height()) for f in detected_faces]


def resize(image, size=(50, 50)):
    if image.shape < size:
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    return cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)


def normalize_intensity(image):
    is_color = len(image.shape) == 3

    if is_color:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return cv2.equalizeHist(image)


def cut_face(image, face):
    (x, y, w, h) = face

    w_rm = int(0.2 * w/2)

    face_croped = image[y:y + h, x + w_rm:x + w - w_rm]
    return face_croped


def draw_rectangle(image, face):
    (x, y, w, h) = face

    w_rm = int(0.2 * w / 2)

    cv2.rectangle(image, (x + w_rm, y), (x + w - w_rm, y + h), (150, 150, 0), 8)


def draw_text(image, coordinates, text):
    cv2.putText(image, text, (coordinates[0], coordinates[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 0), 2)


class Camera:
    def __init__(self, index=0):
        self.video = cv2.VideoCapture(index)

    def __del__(self):
        self.video.release()

    def get_frame(self, in_grayscale=False):
        _, frame = self.video.read()

        if in_grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame

    def show_frame(self, seconds, in_grayscale=False):
        _, frame = self.video.read()

        if in_grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('SnapShot', frame)
        key_pressed = cv2.waitKey(seconds * 1000)

        return key_pressed & 0xFF


def save_model(model):
    joblib.dump(model, config.trained_model, compress=9)


def load_model():
    return joblib.load(config.trained_model)
