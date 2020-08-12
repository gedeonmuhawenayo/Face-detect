import os
import pickle
from operator import itemgetter
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from lib import cfg

np.set_printoptions(precision=2)

training_dir = cfg.output_dir
checkpoints_dir = cfg.checkpoints

if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

label_file = "{}/labels.csv".format(training_dir)
embedding_file = "{}/embeddings.csv".format(training_dir)


print("Loading Face embeddings and labels......")
# load embeddings .......
embeddings = pd.read_csv(embedding_file, header=None).as_matrix()
# load labels ..........
labels = pd.read_csv(label_file, header=None).as_matrix()[:, 1]
labels = map(itemgetter(1), map(os.path.split, map(os.path.dirname, labels)))  # Get the directory.

print("Encoding the labels....")
le = LabelEncoder().fit(labels)
labelsNum = le.transform(labels)
nClasses = len(le.classes_)
print("Training for {} classes.".format(nClasses))

param_grid = {'C': [1, 1e1, 1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]}
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced', probability=True),
                   param_grid)

clf = clf.fit(embeddings, labelsNum)


fName = "{}/classifier.pkl".format(checkpoints_dir)
print("Saving classifier to '{}'".format(fName))
with open(fName, 'w') as f:
    pickle.dump((le, clf), f)
