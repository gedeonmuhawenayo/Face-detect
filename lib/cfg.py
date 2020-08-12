
face_cascade = "lib/model/haarcascade_frontalface_default.xml"
networkModel = "lib/model/openface_nn4.small2.v1.t7"
dlibFacePredictor = "lib/model/shape_predictor_68_face_landmarks.dat"


# Specify the input and output directory here
input_dir = "../dataset/input"
output_dir = "../dataset/train"  # outputs of the embeddings

# Model classifier output
checkpoints = "../checkpoint"

dim = 96
HOST = 127.0.0.1
PORT = 8080