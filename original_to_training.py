import os
import face_detector
import cv2
import config

if __name__ == '__main__':
    images = face_detector.collect_original_images()

    for k, v in images.items():
        person_path = os.path.join(config.training_images_path, k)

        if not os.path.exists(person_path):
            os.mkdir(person_path)

        for img_file in os.listdir(person_path):
            os.remove(os.path.join(person_path, img_file))

        for image in v:
            faces = face_detector.detect_faces_dlib(image[1])

            if len(faces):
                face = faces[0]
                f = face_detector.pipeline(image[1], face)
                cv2.imwrite(os.path.join(person_path, image[0]), f)

