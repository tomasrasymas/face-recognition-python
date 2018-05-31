# Face recognition using Python, OpenCV, dlib, sklear, skimage

Paths:
* images/original - contains photos that are captured running add_person.py
* images/training - contains images ready for recognition. Training images are generated while running original_to_training.py
* models - contains necessary models for perform face recognition and detection


Filter structure

<pre>
images
. original
     . person_1
          . 0.jpg
          . 1.jpg
          . 2.jpg
     . person_2
          . 0.jpg
          . 1.jpg
          . 2.jpg
. training
     . person_1
          . 0.jpg
          . 1.jpg
          . 2.jpg
     . person_2
          . 0.jpg
          . 1.jpg
          . 2.jpg
          
</pre> 
---
Files:
* add_person.py - takes 10 photos of a person and stores photos into images/original/* path
* original_to_training.py - performs original images processing and stores recognition ready faces into images/training/* path
* train_model.py - trains classification model using face in training folder. After execution of this file all data from original folder are processed and moved to training folder. They are ready to be recognized.
* live_recognition.py, recognize_camera.py, recognize_file.py - performs recognition from live stream, from camera photo or from file
* align_dlib.py, face_detector.py, face_recognition.py - functions used for face recognition process
* config.py - configuration parameters
---
## Execution process
1. Add face images using add_person.py
2. Execute original_to_training.py to get training data
3. Execute train_model.py to train model for face recognition
4. Execute one of live_recognition.py, recognize_camera.py, recognize_file.py to test face recognition predictions
