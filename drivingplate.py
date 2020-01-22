import cv2
import imutils
import numpy as np
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
cap = cv2.VideoCapture('video3.mp4') #video3,video6,carlicenseplate - license plate is recognized, when Canny algorithm is used and when the angle of plate's view is none or very little 
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read() 
    frame = cv2.resize(frame, (620,480) )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert to gray scale
    gray = cv2.bilateralFilter(gray, 11, 17, 17) #Blur to reduce noise
    edges = cv2.Canny(gray, 30, 200) #Perform Edge detection

    cnts = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #contours detection
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10] #selecting first 10 values of contours array, which stores the biggest contours
   
    count = None #this variable will contain in the next loop, all detected shapes, which are closed and where aproximated contours of shapes have 4 points
    
#loop over the contours
    for c in cnts:
 #approximate the contour
     peri = cv2.arcLength(c, True)
     approx = cv2.approxPolyDP(c, 0.07 * peri, True) #second parameter of this method determine the shape of detected contours and how many of them will be visible on frame
     
 # if approximated contour has four points, then this is potential plate area
     if len(approx) == 4:
      count = approx
      break

    if count is None:
     detected = 0
     print ("No contour detected")
    else:
     detected = 1

    if detected == 1:
      cv2.drawContours(frame, [count], -1, (0, 255, 0), 3)
      
     
# Masking the part other than the number plate
    mask = np.zeros(gray.shape,np.uint8)
    new_image = cv2.drawContours(mask,[count],0,255,-1,) #this line strongly depends on the quality of captured frame and on the angle of number plate location, when it is not quite clear, it throws error message "(-215:Assertion failed) reader.ptr != NULL in function 'cvDrawContours'"
    new_image = cv2.bitwise_and(frame,frame,mask=mask)

#Cropping area of detected plate
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx+1, topy:bottomy+1]

#Reading the number plate using OCR python library pytesseract
    text = pytesseract.image_to_string(Cropped, config='--psm 11')
    print("Detected Number is:",text)

    cv2.imshow('Single frame',frame)
    cv2.imshow('Number plate',Cropped)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
        break

cap.release()
cv2.destroyAllWindows()
