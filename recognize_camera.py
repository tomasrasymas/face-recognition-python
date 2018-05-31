import face_detector
import cv2

if __name__ == '__main__':
    camera = face_detector.Camera()
    model = face_detector.load_model()

    frame = camera.get_frame()

    detected_faces = face_detector.detect_faces_dlib(frame)

    if detected_faces:
        for face in detected_faces:
            f = face_detector.pipeline(frame, face, features=True)

            prediction = model.predict(f)[0]
            face_detector.draw_text(frame, face, '%s' % prediction)

        cv2.imshow('Face recognition', frame)

        if cv2.waitKey(4000) & 0xFF == 27:
            cv2.destroyAllWindows()
