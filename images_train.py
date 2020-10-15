# import PIL
from PIL import Image
import numpy as np
import cv2
import pickle

import os

face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()



ID = 0
label_ID = {}
xTrain = []
yLabel = []

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

image_dir = os.path.join(BASE_DIR,"images")

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
            label = os.path.basename(root).replace(" ", "-").lower()



            if not label in label_ID:
                label_ID[label] = ID
                ID += 1
            ID_ = label_ID[label]

# convert into grayscale
            pil =Image.open(path).convert("L")

# convert into numpy array
            image_array = np.array(pil, "uint8")


            faces = face.detectMultiScale(image_array, scaleFactor=1.1, minNeighbors=6)

            for (x, y, w, h ) in faces:
                roi = image_array[y:y+h, x:x+w]
                xTrain.append(roi)
                yLabel.append(ID)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ID, f)

recognizer.train(xTrain, np.array(yLabel))
recognizer.save("Train.yml")