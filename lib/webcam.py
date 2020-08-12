import numpy as np
import cv2
import time
from scipy import misc
import urllib2
import pickle
import os
from lib.utils import load_lib
import requests
import json
import grequests
import threading
from lib import cfg


def open_door():
    urls = ["http://10.22.0.112:5000/open"]
    rs = (grequests.get(u) for u in urls)
    grequests.map(rs)
    return


def check_staff(uid):
    return uid in staffs


def check_time():
    return True


face_cascade = cfg.face_cascade
networkModel = cfg.networkModel
dlibFacePredictor = cfg.dlibFacePredictor
model = "{}/classifier.pkl".format(cfg.checkpoints)  # pkl file
output = cfg.output_dir
dim = cfg.dim
HOST = cfg.HOST
PORT = cfg.PORT

if not os.path.exists(output):
    os.makedirs(output)

with open(model, 'r') as f:
    (le, clf) = pickle.load(f)

# Start the webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# initialize lib
lib = load_lib(face_cascade=face_cascade, network_model=networkModel,
               dlib_face_predictor=dlibFacePredictor)

url = cfg.url_staff_list
staff_info_json = requests.get(url)
staff_info = json.loads(staff_info_json.text)
staffs = []
for staff in staff_info['staffs']:
    staffs += [staff['uid']]

start = time.time()
startOpen = time.time()
name = "abc"
nameopen = ""
counttime = 0

while True:
    ret, frame = cap.read()
    rgbImg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    end = time.time()
    if end - startOpen < 2 and nameopen != "":
        cv2.putText(frame, str(nameopen) + "  Openning door...",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)

    if end - start < 0.1:
        cv2.imshow('frame', frame)
        continue
    else:
        counttime += 1
        start = end

    faces = lib.face_cascade.detectMultiScale(rgbImg, 1.3, 5)

    for (x, y, w, h) in faces:
        sub_face = rgbImg[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        path = "test_" + str(time.time()) + ".png"
        face_align = lib.align_function(np.array(sub_face))

        if face_align is not None:
            vector = lib.get_vector(face_align).reshape(1, -1)
            predictions = clf.predict_proba(vector).ravel()
            buffer = sorted(predictions, reverse=True)

            if counttime >= 10:
                misc.imsave(output + "/" + str(time.time()) + ".png", face_align)
                counttime = 0
            for j in range(len(predictions)):
                if predictions[j] == buffer[0]:
                    name = le.inverse_transform(j)
                    print(name)
                    cv2.putText(frame, "Predict {} with {:.2f} confidence.".format(name, buffer[0]),
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                cv2.LINE_AA)
            if (check_staff(name)) & (check_time()) & (buffer[0] > 0.7):
                endOpen = time.time()
                if endOpen - startOpen > 5.0:
                    try:
                        urllib2.urlopen("http://10.22.0.25/store.php?ldap=" + name).read()
                    except ValueError:
                        print("Error in the URL ")
                    pass

                    print()
                    cv2.putText(frame, str(name) + " Openning door...", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    nameopen = str(name)
                    startOpen = time.time()
                    t = threading.Thread(target=open_door)
                    t.start()
                break

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
