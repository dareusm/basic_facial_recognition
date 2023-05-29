#Facial Recognition App
import cv2

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)

while(True):
    
    successful_frame_read, frame = webcam.read()
    
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow("Face Detector", frame)
    
    cv2.waitKey(1)

"""
#Choose an image to detect faces in
img = cv2.imread('the_rock.png')

#Convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detect Faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
#Draw rectangles around the faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
    
#Display the image with the faces
cv2.imshow('Face Detector', img)


cv2.waitKey()
"""



print('Code Complete')

