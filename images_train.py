# import PIL
from PIL import Image
import numpy as np
import cv2
import pickle

import os

face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


recognizer = cv2.face.LBPHFaceRecognizer_create()


# current ID for face 
ID_ = 1
# create a dictionary for the labels 
label_ID = {}
# Save images in array 
xTrain = []
yLabel = []





# Wherever its directory is saved look for path 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Walk through files and look for jpg or png files
image_dir = os.path.join(BASE_DIR,"images")

# Find the path in the directory for root  
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
# Replace spaces with dash and lowercase the labels names 
            label = os.path.basename(root).replace(" ", "-").lower()


# loop for the label ID to be equal to the value being used This creates the dictionary of labels 
            if not label in label_ID:
                label_ID[label] = ID_
                ID_ += 1
            ID_ = label_ID[label]

# Get image of the path convert into gray scale 
            pil =Image.open(path).convert("L")

# convert into numpy array uint8 as the type (numbers)
            image_array = np.array(pil, "uint8")

# use face cascade to grab the faces multi scale same as we did in file main 
            faces = face.detectMultiScale(image_array, scaleFactor=1.1, minNeighbors=6)

            for (x, y, w, h ) in faces:
                roi = image_array[y:y+h, x:x+w]
                xTrain.append(roi)
                yLabel.append(ID_)


# writing bytes as file in pickle dump IDs in file 
with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ID, f)

# put training data into numpy arrays and convert y labels into numpy arrays 
recognizer.train(xTrain, np.array(yLabel))
recognizer.save("Train.yml")
