import face_detector
from sklearn import svm


if __name__ == '__main__':
    data, labels = face_detector.collect_training_dataset()

    data = [face_detector.image_to_features(d) for d in data]

    model = svm.SVC(kernel='linear')
    model.fit(data, labels)

    face_detector.save_model(model)

