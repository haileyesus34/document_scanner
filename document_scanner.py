import cv2 
import numpy as np
from utils import initializeTrackbacks, valTrackbars, biggest_conout, draw_rectangle, reorder
import matplotlib.pyplot as plt 

width = 640
height = 480

img = cv2.imread('\\Users\\hanna m\\machinelearning\\deep_learning\\cv\\document_scanner\\doc6.jpg')
img = cv2.resize(img,(640, 420))
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

initializeTrackbacks()

while True:
  imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
  threshold = valTrackbars()
  imgthresh = cv2.Canny(imgBlur, threshold[0], threshold[1])
  kernel = np.ones((5,5))
  imgDilate = cv2.dilate(imgthresh, kernel, iterations=2)
  imgErode = cv2.erode(imgDilate, kernel, iterations=1)

  # find all contours
  imgContours = img.copy()
  imgBigContours = img.copy()
  contours, _ = cv2.findContours(imgErode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  image = cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 4)

  #cv2.imshow('image', image)
  biggest, maxAread = biggest_conout(contours)
  
  if biggest.size != 0:
      biggest = reorder(biggest)
      for i in biggest:
         pic = cv2.circle(imgBigContours, (i[0][0], i[0][1]), 10 , (0, 255, 0), -1)
      

      pts1 = np.float32(biggest)
      pts2 = np.float32([[[0,0]], [[width, 0]], [[0, height]], [[width, height]]])
      matrix = cv2.getPerspectiveTransform(pts1, pts2)
      scanned_image = cv2.warpPerspective(img, matrix, (480, 480))

      #scanned_image = cv2.resize(scanned_image, (width, height))

      # apply adaptive threshold
      scanned_gray = cv2.cvtColor(scanned_image, cv2.COLOR_BGR2GRAY)
      scanned_gray_thresh = cv2.adaptiveThreshold(scanned_gray, 255, 1, 1,3, 2)
      scanned_thresh_not = cv2.bitwise_not(scanned_gray_thresh)
      scanned_thresh_not = cv2.medianBlur(scanned_thresh_not, 3)
   

      images = [imgGray, imgBlur, imgthresh, imgDilate, imgErode, scanned_thresh_not]
      labels = ['Gray', 'blur', 'thresh', 'Dilate', 'Erode', 'scanned_image_thresh']
   
      cv2.imshow('scanned _image', scanned_image)
      cv2.imshow('scanned_thresh_not', scanned_thresh_not)

  if cv2.waitKey(27) & 0xff == ord('q'):
    break 

   
cv2.destroyAllWindows()