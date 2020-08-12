import numpy as np
import os
from lib.utils import load_lib
import glob
from PIL import Image
from lib import cfg

# ------ Configurations ------- #
face_cascade = cfg.face_cascade
networkModel = cfg.networkModel
dlibFacePredictor = cfg.dlibFacePredictor
input_dir = cfg.input_dir
output_dir = cfg.output_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

lib = load_lib(face_cascade=face_cascade, network_model=networkModel,
               dlib_face_predictor=dlibFacePredictor)

f1 = open(output_dir+'/labels.csv', 'w')
f2 = open(output_dir+'/embeddings.csv', 'w')


list_file = glob.glob(input_dir+"/*/*")
array = []
index = []

for i in range(len(list_file)):
    folder = list_file[i].split("/")
    file = folder[len(folder) - 1]
    name = folder[len(folder) - 2]
    flag = True
    num = 0
    for j in range(len(array)):
        if array[j] == name:
            num = index[j]
            flag = False
            break
    if flag:
        num = len(index) + 1
        array.append(name)
        index.append(num)
        print(name)

    f1.write(str(num)+","+name+"/"+file+"\n")
    vector = lib.get_vector(np.asarray(Image.open(list_file[i])))
    # vector.flatten()
    for_write = ""
    for a in vector:
        for_write += str(a) + ","
    f2.write(for_write[0:len(for_write) - 1]+"\n")

print(len(array))
print(len(index))

f1.close()
f2.close()
