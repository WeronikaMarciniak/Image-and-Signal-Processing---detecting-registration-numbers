import numpy as np
import cv2

blur=21   # bluring value

cap = cv2.VideoCapture('video6.mp4') #video3,video6,carlicenseplate - license plate is recognized, when Canny algorithm is used and when the angle of plate's view is none or very little 
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read() 
  # Our operations on the frame come here
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #converting to gray scale
    eq = cv2.equalizeHist(gray) #histogram equalization
    frame = cv2.GaussianBlur(frame, (blur,blur), 0, 0) #Gaussian Blur
    frame = cv2.Canny(eq,20,60,1)  # Canny function
   
    cv2.imshow('Video',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
        break

cap.release()
cv2.destroyAllWindows()


