# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 20:29:45 2020

@author: elect
"""
import cv2
import numpy as np
import my_utils
from PIL import Image
import pytesseract
from pytesseract import Output
import random
import table_detection as table


#WIDTH AND HEIGHT OF THE PREVIEWS IN THE MAIN WINDOW 
fixedheightImg = 480 
fixedwidthImg = 320



print(cv2.getBuildInformation())
webCamFeed = False
pathImage = "ejemplos/tabla1_numeros.jpg"
cap = cv2.VideoCapture(0)
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

#CHECK IF CAMERA IS NOT SET AS INPUT SOURCE WE OBTAIN THE WIDTH AND HEIGHT FROM THE pathImage ELSE WE SET A STATIC WIDTH AND HEIGHT
if not webCamFeed:
    image = Image.open(pathImage)
    widthImg, heightImg = image.size
else:
    widthImg = 720
    heightImg = 1280
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, widthImg)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, heightImg)
#WE INITIALIZE A WINDOW WITH TRACKBARS TO MODIFY IN REALTIME THE PREPROCESING CONFIGURATION
my_utils.initializeTrackbars()
count = 0
while True:
    if webCamFeed:success, img = cap.read()
    else:img = cv2.imread(pathImage)
    img = cv2.resize(img, (widthImg, heightImg)) # RESIZE IMAGE
    imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # ADD GAUSSIAN BLUR
    thres=my_utils.valTrackbars() # GET TRACK BAR VALUES FOR THRESHOLDS
    imgThreshold = cv2.Canny(imgBlur,thres[0],thres[1]) # APPLY CANNY BLUR
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2) # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION

    ## FIND ALL COUNTOURS
    imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # DRAW ALL DETECTED CONTOURS


    # FIND THE BIGGEST COUNTOUR
    biggest, maxArea = my_utils.biggestContour(contours) # FIND THE BIGGEST CONTOUR
    if biggest.size != 0:
        biggest=my_utils.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
        imgBigContour = my_utils.drawRectangle(imgBigContour,biggest,2)
        pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
        
        nws = cv2.minAreaRect(biggest)
        (xws, yws), (widthws, heightws), anglesws = nws
        
        #print("NWS ES:")
        #print(nws)
        imgWarpCutSize = cv2.resize(imgWarpColored, (round(widthws),round(heightws)))
        #REMOVE 20 PIXELS FORM EACH SIDE
        #imgWarpColored=imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        #imgWarpColored = cv2.resize(imgWarpColored,(widthImg,heightImg))

        # APPLY ADAPTIVE THRESHOLD
        imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre= cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre=cv2.medianBlur(imgAdaptiveThre,3)
        #if imgWarpColored
        

        # Image Array for Display
        imageArray = ([ img,
                        imgGray,
                        imgThreshold,
                        imgContours],
                      [ imgBigContour,
                        imgWarpColored, 
                        imgWarpGray,
                        imgAdaptiveThre])

    else:
        imageArray = ([img,
                       imgGray,
                       imgThreshold,
                       imgContours],
                      [imgBlank,
                       imgBlank,
                       imgBlank,
                       imgBlank])
        
    
    #cv2.imwrite("scanned/Original"+str(count)+".jpg",img)
    #cv2.imwrite("scanned/Warp-Prespective"+str(count)+".jpg",imgWarpColored)
    # LABELS FOR DISPLAY
    lables = [["Original","Gray","Threshold","Contours"],
              ["Biggest Contour","Warp Prespective","Warp Gray","Adaptive Threshold"]]
    desiredWidth = 1300/4
    desiredHeight = 768/2
    
    desiredWidth = desiredWidth/imageArray[0][0].shape[1]
    desiredHeight = desiredHeight/imageArray[0][0].shape[0]
    
    proportion = min((desiredWidth,desiredHeight))
    
    stackedImage = my_utils.stackImages(imageArray,proportion,lables)
    cv2.imshow("RECORTE Y PREPROCESAMIENTO",stackedImage)
    table.processTables(imgWarpCutSize,count)
   
    
    