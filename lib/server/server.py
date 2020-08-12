from flask import Flask
from flask import request
import pickle
import time
from scipy import misc
import os
from lib.utils import load_lib
from lib import cfg

face_cascade = cfg.face_cascade
networkModel = cfg.networkModel
dlibFacePredictor = cfg.dlibFacePredictor
dim = cfg.dim
HOST = cfg.HOST
PORT = cfg.PORT

output = "output"
if not os.path.exists(output):
    os.makedirs(output)

lib = load_lib(face_cascade=face_cascade, network_model=networkModel, dlib_face_predictor=dlibFacePredictor)

app = Flask(__name__)


@app.route('/', methods=['POST'])
def save():
    data = request.stream.read()
    if data is not None:
        face_align = pickle.loads(data)
        path = "test_"+str(time.time())+".png"
        print(path)
        misc.imsave(output + "/" + path, face_align)
        vector = lib.get_vector(face_align)
        lib.save_redis(path, vector)
    return "OK"


if __name__ == "__main__":
    app.run(host=HOST, port=PORT)
