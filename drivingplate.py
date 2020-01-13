
#import numpy as np
import cv2
#import matplotlib.pyplot as plt

video = cv2.VideoCapture('CarLicensePlate.mp4')

while(video.isOpened()):
    ret, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()


