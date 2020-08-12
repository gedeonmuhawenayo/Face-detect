import pickle
import numpy as np
import sys
from PIL import Image
from lib.utils import load_lib
from lib import cfg

face_cascade = cfg.face_cascade
networkModel = cfg.networkModel
dlibFacePredictor = cfg.dlibFacePredictor
dim = cfg.dim
model = "{}/classifier.pkl".format(cfg.checkpoints)  # pkl file
img = sys.argv[1]  # test image

# initialize
lib = load_lib(face_cascade=face_cascade, network_model=networkModel,
               dlib_face_predictor=dlibFacePredictor)

with open(model, 'r') as f:
    (le, clf) = pickle.load(f)

print("\n===== {} =====".format(img))
im = np.asarray(Image.open(img))

list_face = lib.face_detect(im)
if len(list_face) == 0:
    print("========== No face detected ======")

for i in range(len(list_face)):
    face_align = lib.align_function(np.array(list_face[i]))
    if face_align is not None:
        rep = lib.get_vector(face_align).reshape(1, -1)
        predictions = clf.predict_proba(rep).ravel()
        buffer = sorted(predictions, reverse=True)
        count = 1
        for top_3 in buffer:
            if count > 3:
                break
            for j in range(len(predictions)):
                if predictions[j] == top_3:
                    print("Predict {} with {:.2f} confidence.".format(le.inverse_transform(j), top_3))
                    break
            count = count + 1
