import cv2

#load trained cascade classifier
face_cascade =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#read the given image
color_image =cv2.imread('mr.jpg')

#convert color image into grayscale
gray_image =cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)

#detect faces ROI
#syntax Classifier, detectmultiscale(input image,scale factor,min neighbour)
faces =face_cascade.detectmultiscale(gray_image,1.1, 5)

#draw rectangle around the faces 
for(x,y,w,h) in faces:
    cv2.rectangle(color_image,(x,y),(x+w,y+h),(0,255,0),4)

#show image
cv2.imshow('image',color_image)

#wait to close window
cv2.waitKey()

#close all window
cv2.destroyAllWindow()