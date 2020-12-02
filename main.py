# Facial Recognition 
# CS 4200 Fall 2020
# Darius Wortham & Michael Prothero

# This program will Recognize Faces using algorithms to detect 
# your face in Real-Time 
# After Detecting a face, the Algorith will identify who that person(face) is.

# To quit press 'q' on keyboard 



import numpy as np
import cv2
import sys
import os
import face_recognition
import pickle #for converting python objects 


cascPath = sys.argv[0]

# Get a reference to the Video Camera
cap = cv2.VideoCapture(0)

#xml file that contains Face algorithm for facial recognition
# face = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# human-readable structured data format
recognizer.read("train.yml")

the_labels = {"mike": 1}
with open("labels.pickle", 'rb') as f:
    the_labels_one = pickle.load(f)

    
# invert the labels 
    the_labels = {v:k for k,v in the_labels.items()}



# capture the frame of the video, frame by frame
while True:

# Grab a single frame from the video
    ret, frame = cap.read()




# first Convert the frame to grey color so facial recocognition can read video
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)



# Draw a box around the face
    for (x, y, w, h) in faces:
        # print(x, y, w, h)
        
# second find the region of interest of the face being used 
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        img_item = "my-image.png"


# recognize the image
        ID_, conf = recognizer.predict(roi_gray)
        if conf >= 45: #confidence
            print(ID_)
            print(the_labels[ID_])
            
# put text on box name of the labels 
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = the_labels[ID_]
            
# color of box draw the box
            color = (255, 255, 255)
            stroke = 1
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

# images read as numpy
        cv2.imwrite(img_item, roi_gray)

        color = (255, 0, 0)
        stroke = 2
        width = x + w
        height = y + h

# cv2.rectangle(frame, (x, y), (width, height), color, stroke)
# draw the rectangle / color and stroke of box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)



# font = cv2.FONT_HERSHEY_DUPLEX

# Display the image and hit p on the keyboard to quit
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('p'):
        break

cap.release()
cv2.destroyAllWindows()

# End of file

