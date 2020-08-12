# Face-detect
A simple face detection system for door management. 

## Some Housekeeping
Before getting started running this code, the following instructions should be done

+ Download the `haarcascade_frontalface_default.xml` file
+ Download the `openface_nn4.small2.v1.t7` model
+ Download the `shape_predictor_68_face_landmarks.dat` model
+ Look at the configuration file and include your necessary paths

## To Run this software
+ Get the embeddings of your dataset by running `./embeddings.sh`
+ Train your model by running `./train.sh`
+ Test your trained model by running `./test.sh`
+ Deploy with webcam for opening door using `./webcam.sh`

