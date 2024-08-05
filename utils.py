import cv2 
import numpy as np 

def nothing(x):
    pass

def initializeTrackbacks():
    cv2.namedWindow('TrackBars')
    cv2.resizeWindow('TrackBars', (360, 240))
    cv2.createTrackbar('Threshold1', 'TrackBars', 200, 255, nothing)
    cv2.createTrackbar('Threshold2', 'TrackBars', 200, 255, nothing)

def valTrackbars():
    threshold1 = cv2.getTrackbarPos('Threshold1', 'TrackBars')
    threshold2 = cv2.getTrackbarPos('Threshold2', 'TrackBars')
    src = threshold1, threshold2

    return src

def biggest_conout(contours):
    biggest = np.array([])
    max_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02*peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area

def reorder(myPoints):
   newPoints = np.zeros((4,1,2), np.int32)
   add = myPoints.sum(2)
   myPoints = myPoints.tolist()
   newPoints[0] = myPoints[np.argmin(add)]
   newPoints[3] = myPoints[np.argmax(add)]
   myPoints.remove(myPoints[np.argmax(add)])
   myPoints.remove(myPoints[np.argmin(add)])
   

   diff = np.diff(myPoints, axis= 0)

   if diff[0][0][1] < 0:
      newPoints[1] = myPoints[1]
      newPoints[2] = myPoints[0]
   else: 
      newPoints[1] = myPoints[0]
      newPoints[2] = myPoints[1]
    
   return newPoints

def draw_rectangle(img, biggest, t):

    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), t)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), t)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), t)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), t)

