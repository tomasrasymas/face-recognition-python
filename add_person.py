import os
import cv2
import face_detector
import config


if __name__ == '__main__':
    camera = cv2.VideoCapture(0)

    cv2.namedWindow("preview")

    person_name = input('Person name: ').lower()

    person_folder = os.path.join(config.original_images_path, person_name)

    if not os.path.exists(person_folder):
        os.mkdir(person_folder)
        counter = 0
        timer = 0

        while counter < config.number_of_faces and camera.isOpened():
            ret, frame = camera.read()

            faces = face_detector.detect_faces_dlib(frame)

            if len(faces):
                face = faces[0]

                if timer % 200 == 50:
                    cv2.imwrite(os.path.join(person_folder, '%s.jpg' % counter), frame)
                    counter += 1

                face_detector.draw_text(frame, face, str(counter))
                face_detector.draw_rectangle(frame, face)
                cv2.imshow('Camera image', frame)

            if cv2.waitKey(20) & 0xFF == 27:
                break

            timer += 50

    camera.release()

    cv2.destroyAllWindows()