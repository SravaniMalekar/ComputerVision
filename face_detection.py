import numpy as np
import cv2 

#change path depending upon where your image is present
denis = cv2.imread('../DATA/Denis_Mukwege.jpg',0)
solvay = cv2.imread('../DATA/solvay_conference.jpg',0)

face_cascade = cv2.CascadeClassifier('../DATA/haarcascades/haarcascade_frontalface_default.xml')
def adj_detect_face(img):
    
    face_img = img.copy()
  
    face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.2, minNeighbors=5) 
    
    for (x,y,w,h) in face_rects: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 10) 
        
    return face_img

result = adj_detect_face(solvay)
result2= adj_detect_face(denis)

while True:
    
    # SHow the 2 windows
    cv2.imshow('face', result)
    cv2.imshow('face2', result2)
    k = cv2.waitKey(1)

    if k == 27:
        break
        
    # Close everything if Esc is pressed
    k = cv2.waitKey(1)

    if k == 27:
        break
        
cv2.destroyAllWindows()
        
