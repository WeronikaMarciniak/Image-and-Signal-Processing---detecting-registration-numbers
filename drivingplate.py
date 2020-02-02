import sys
import cv2
import imutils
import numpy as np
import pytesseract
import re
# import time

def order_points(pts):
    # initialzie a list of coordinates that will be ordered such that the first entry
    # in the list is the top-left, the second entry is the top-right, the third
    # is the bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas the bottom-right point
    # will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # now, compute the difference between the points, the top-right point will have
    # the smallest difference, whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # compute the width of the new image, which will be the maximum distance between
    # bottom-right and bottom-left x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # compute the height of the new image, which will be the maximum distance between
    # the top-right and bottom-right y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # now that we have the dimensions of the new image, construct the set of destination points
    # to obtain a "birds eye view", (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    # return the warped image
    return warped

# ===================================================

if len(sys.argv) != 2:
    print("usage: %s <video_name.mp4>" % sys.argv[0])
else:
    
    text_old = ""
    cap = cv2.VideoCapture(sys.argv[1])

    while(True):
        # capture frame-by-frame
        ret, frame = cap.read() 
        if ret == 0:
            break
        else:
            # resize frame
            # frame = cv2.resize(frame, (620, 480))
            # frame = cv2.resize(frame, (640, 360))
            
            # convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # blur to reduce noise
            gray = cv2.bilateralFilter(gray, 11, 19, 19)
            # cv2.imshow('Gray', gray)
            
            # perform Edge detection
            edges = cv2.Canny(gray, 30, 250)
            # cv2.imshow('Edges', edges)
            
            # contours detection
            cnts = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            
            # selecting first 10 values of contours array, which stores the biggest contours
            cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
            
            # in 2nd loop this variable will contain all detected shapes that are closed and where approximated contours of shapes have 4 points
            cont = None
            
            # loop over the contours and approximate the contour
            for c in cnts:
                # contour perimeter
                perimeter = cv2.arcLength(c, True)
                
                # epsilon is maximum distance from contour to approximated contour. It is an accuracy parameter. A wise selection of epsilon is needed to get the correct output.
                epsilon = 0.01 * perimeter # 0.01
                approx = cv2.approxPolyDP(c, epsilon, True)
                
                # if approximated contour has four points, then this is potential plate area
                if len(approx) == 4:
                    cont = approx
                    break
            
            if cont is None:
                # print ("No contour detected")
                continue
            else:
                cv2.drawContours(frame, [cont], -1, (0, 255, 0), 2)
                cv2.imshow('Single frame', frame)
                
                # OLD VERSION of the code didn't use transformation and just cropped
                # the plate. Now this block of code is not necesarry anymore,
                # but didn't want to remove it.
                
                # masking the part other than the number plate
                # mask = np.zeros(gray.shape,np.uint8)
                
                # new_image = cv2.drawContours(mask,[cont],0,255,-1,)
                # new_image = cv2.bitwise_and(frame,frame,mask=mask)
                
                # cv2.imshow('New image', new_image)
                
                # cropping area of detected plate
                # (x, y) = np.where(mask == 255)
                # (topx, topy) = (np.min(x), np.min(y))
                # (bottomx, bottomy) = (np.max(x), np.max(y))
                # cropped = gray[topx-5:bottomx+5, topy-5:bottomy+5]
                
                # cv2.imshow('Cropped plate number', cropped)
                
                # get points from contour
                rectangle = cv2.minAreaRect(cont)
                box = cv2.boxPoints(rectangle)
                
                ptstmp = []
                
                for p in box:
                    pt = [p[0], p[1]]
                    ptstmp.append(pt)
                
                # make a numpy array from the points
                pts = np.array(ptstmp, dtype = "float32")
                
                transformed = four_point_transform(gray, pts)
                cv2.imshow('Plate number', transformed)
                
                # reading the number plate using OCR python library, "pytesseract"
                text = pytesseract.image_to_string(transformed, lang='pol', config='--psm 13')
                text = text.replace(':', ' ')
                text = text.replace('=', ' ')
                text = text.replace('-', ' ')
                
                # make one space if many are next to each other
                text = re.sub(' +',' ', text)
                
                # make some contraints on the text length, don't show the text if previous was the same, match regexp
                if len(text) > 5 and len(text) < 10 and text != text_old and re.match('^[A-Z0-9 ]*$', text):
                    print("Detected Number is: ", text)
                
                if len(text) > 5 and len(text) < 10 and re.match('^[A-Z0-9 ]*$', text):
                   text_old = text
                
                # needed some time to take screenshots :)
                # time.sleep(2)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
                    break
    
    cap.release()
    cv2.destroyAllWindows()
