import face_detector
import cv2
import os

if __name__ == '__main__':
    model = face_detector.load_model()

    frame = cv2.imread(os.path.join('images', 'original', 'albertas', '0.jpg'), 1)

    detected_faces = face_detector.detect_faces_dlib(frame)

    if detected_faces:
        for face in detected_faces:
            f = face_detector.pipeline(frame, face, features=True)

            prediction = model.predict(f)[0]
            face_detector.draw_text(frame, face, '%s' % prediction)

        cv2.imshow('Face recognition', frame)

        if cv2.waitKey(5000) & 0xFF == 27:
            cv2.destroyAllWindows()
