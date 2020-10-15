import numpy as np
import cv2
import sys
import os
import face_recognition
import pickle



cascPath = sys.argv[0]


# Get a reference to the Video Camera
cap = cv2.VideoCapture(0)


# face = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("train.yml")

the_labesl = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    labels = pickle.load(f)
    labels = {v:k for k,v in the_labesl.items()}





# Import sample pictures and learn how recognize it






# Create a array of known face encodings






# Initialize Variables of FaceLocation, FaceEncoding, FaceNames, Process the frame















# capture the frame of the video, frame by frame
while True:

# Grab a single frame from the video
    ret, frame = cap.read()




# Convert the frame to grey color so facial recocognition can read video
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)



#Draw a box around the face
    for (x, y, w, h) in faces:
        # print(x, y, w, h)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        img_item = "my-image.png"


# recognize the image
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45:
            print(id_)
            print(labels[id_])

        cv2.imwrite(img_item, roi_gray)

        color = (255, 0, 0)
        stroke = 2
        width = x + w
        height = y + h

# cv2.rectangle(frame, (x, y), (width, height), color, stroke)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)



# font = cv2.FONT_HERSHEY_DUPLEX

# Display the image and hit p on the keyboard to quit
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('p'):
        break

cap.release()
cv2.destroyAllWindows()

# End of file

