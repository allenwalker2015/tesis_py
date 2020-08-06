# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 09:08:55 2020

@author: elect
"""


import cv2
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import sys
import random
import my_utils

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract


#WIDTH AND HEIGHT OF THE PREVIEWS IN THE MAIN WINDOW 
fixedheightImg = 320
fixedwidthImg = 480

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
        
    return (cnts, boundingBoxes)
def processTables(image,count):
    
    img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #thresholding the image to a binary image
    #thresh,img_bin = cv2.threshold(img,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #imgWarpGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #img_bin= cv2.adaptiveThreshold(img, 255, 1, 1, 7, 2)
    img_bin= cv2.adaptiveThreshold(~img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,91, -2)
    #img_bin = cv2.bitwise_not(img_bin)
    horizontal = img_bin.copy()
    vertical = img_bin.copy()
	# Specify size on horizontal axis
    scale = my_utils.valScale()
    #print(horizontal.shape[0])
    #print("X")
    #print(horizontal.shape[1])
    
    horizontalsize = int(horizontal.shape[1]/scale)
	# Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT,(horizontalsize, 1))
    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
    horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))
	#dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1)); // expand horizontal lines
	#Show extracted horizontal lines
    #imshow("horizontal", horizontal)
	#Specify size on vertical axis
    verticalsize = int(vertical.shape[0] / scale)
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT,(1, verticalsize))
	#Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
    #dilate(vertical, vertical, verticalStructure, Point(-1, -1)); // expand vertical lines
	# Show extracted vertical lines
    #imshow("vertical", vertical);

    # create a mask which includes the tables
    mask = horizontal + vertical
    newsize = vertical.shape[0]/vertical.shape[1]
    #print(newsize)
    cv2.imwrite("scanned/mask"+str(count)+".jpg",mask)
    #maskRes = cv2.resize(mask, ( 800,round(newsize*800)))
    #cv2.imshow("TABLE MASK", maskRes)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img_vh = cv2.addWeighted(vertical, 0.5, horizontal, 0.5, 0.0)#Eroding and thesholding the image
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    thres=my_utils.valTableTrackbars()
    thresh, img_vh = cv2.threshold(img_vh,thres[0],thres[1], cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bitxor = cv2.bitwise_xor(img,img_vh)
    bitnot = cv2.bitwise_not(bitxor)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    print(hierarchy) 
    #image = cv2.drawContours(image, contours, -1, (0,255,0), 3)
    #joints = cv2.bitwise_and(horizontal, vertical)
    #jointsRes = cv2.resize(joints, ( 800,round(newsize*800)))
    #cv2.imshow("JOINTS",jointsRes)
    #contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    imgWithContours = img.copy()
    key = cv2.waitKey(1)
    if key == ord('q'):
        cv2.destroyAllWindows()
        sys.exit()
    if key == ord('c'):
        # cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),(1100, 350), (0, 255, 0), cv2.FILLED)
        #cv2.putText(stackedImage, "Se esta procesando la tabla!", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        #cv2.imshow("PROCESAMIENTO DE TABLA",stackedImage)
        # Sort all the contours by top to bottom.
        #contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")
        
        #Creating a list of heights for all detected boxes
        #heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
        
        #Get mean of heights
        #mean = np.mean(heights)
        
        #Create list box to store all boxes in  
        box = []
        # Get position (x,y), width and height for every contour and show the contour on image
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if (w<2000 and h<700):
                #print("box1:"+x+','+w+','+y+','+h)
                image = cv2.rectangle(image,(x,y),(x+w,y+h),random_color(),-1)
                box.append([x,y,w,h])
                cv2.imshow("CELL DETECTED",img[y:y+h, x:x+w])
                cv2.waitKey(0)
            else:
                
                image = cv2.rectangle(image,(x,y),(x+w,y+h),random_color(),3)
                print("box1:"+str(x)+','+str(y)+','+str(w)+','+str(h))
                print("Esto no es celda")
        imageRsz =  cv2.resize(image, ( 800,round(newsize*800)))  
        cv2.imwrite("scanned/cells"+str(count)+".jpg",image)
        cv2.imshow("CELLS",imageRsz)
        
    # SAVE IMAGE WHEN 's' key is pressed
    if key == ord('a'):
       # cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),(1100, 350), (0, 255, 0), cv2.FILLED)
        #cv2.putText(stackedImage, "Se esta procesando la tabla!", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        #cv2.imshow("PROCESAMIENTO DE TABLA",stackedImage)
        # Sort all the contours by top to bottom.
        contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")
        
        #Creating a list of heights for all detected boxes
        heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
        
        #Get mean of heights
        mean = np.mean(heights)
        
        #Create list box to store all boxes in  
        box = []
        # Get position (x,y), width and height for every contour and show the contour on image
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if (w<2000 and h<700 and w>10):
                #print("box1:"+x+','+w+','+y+','+h)
                image = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
                box.append([x,y,w,h])
                cv2.imshow("CELL DETECTED",img[y:y+h, x:x+w])
                cv2.waitKey(0)
                
        plotting = plt.imshow(image,cmap='gray')
        plt.show()
        
        #Creating two lists to define row and column in which cell is located
        row=[]
        column=[]
        j=0
        print("Ordenando las cajas por filas y columnas....")
        #Sorting the boxes to their respective row and column
        for i in range(len(box)):    
                
            if(i==0):
                column.append(box[i])
                previous=box[i]    
            
            else:
                if(box[i][1]<=previous[1]+mean/2):
                    column.append(box[i])
                    previous=box[i]            
                    
                    if(i==len(box)-1):
                        row.append(column)        
                    
                else:
                    row.append(column)
                    column=[]
                    previous = box[i]
                    column.append(box[i])
                    
        print(column)
        print(row)
        
        print("Calculando el numero de celdas....")
        #calculating maximum number of cells
        countcol = 0
        for i in range(len(row)):
            countcol = len(row[i])
            if countcol > countcol:
                countcol = countcol
        
        #Retrieving the center of each column
        center = [int(row[i][j][0]+row[i][j][2]/2) for j in range(len(row[i])) if row[0]]
        
        center=np.array(center)
        center.sort()
        print(center)
        #Regarding the distance to the columns center, the boxes are arranged in respective order
        
        finalboxes = []
        for i in range(len(row)):
            lis=[]
            for k in range(countcol):
                lis.append([])
            for j in range(len(row[i])):
                diff = abs(center-(row[i][j][0]+row[i][j][2]/4))
                minimum = min(diff)
                indexing = list(diff).index(minimum)
                lis[indexing].append(row[i][j])
            finalboxes.append(lis)
        
        print("Reconociendo el texto en cada una de las celdas....")
        #from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
        outer=[]
        for i in range(len(finalboxes)):
            for j in range(len(finalboxes[i])):
                inner=''
                if(len(finalboxes[i][j])==0):
                    outer.append(' ')
                else:
                    for k in range(len(finalboxes[i][j])):
                        y,x,w,h = finalboxes[i][j][k][0],finalboxes[i][j][k][1], finalboxes[i][j][k][2],finalboxes[i][j][k][3]
                        finalimg = bitnot[x:x+h, y:y+w]
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                        border = cv2.copyMakeBorder(finalimg,2,2,2,2, cv2.BORDER_CONSTANT,value=[255,255])
                        resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                        dilation = cv2.dilate(resizing, kernel,iterations=1)
                        erosion = cv2.erode(dilation, kernel,iterations=2)
                        
                        out = pytesseract.image_to_string(erosion,lang='spa')
                        if(len(out)==0):
                            out = pytesseract.image_to_string(erosion, config='--psm 3',lang='spa')
                        inner = inner +" "+ out
                    outer.append(inner)
        
        #Creating a dataframe of the generated OCR list
        arr = np.array(outer)
        dataframe = pd.DataFrame(arr.reshape(len(row), countcol))
        print(dataframe)
        data = dataframe.style.set_properties(align="left")
        #Converting it in a excel-file
        data.to_excel("scanned/output.xlsx")
        
    #my_utils.drawRectangle(imgBigContour,biggest,2)
    imageArray = ([cv2.resize(img,(fixedwidthImg,fixedheightImg)),
                       cv2.resize(img_bin,(fixedwidthImg,fixedheightImg)),
                        cv2.resize(vertical,(fixedwidthImg,fixedheightImg)),
                        cv2.resize(horizontal,(fixedwidthImg,fixedheightImg))],
                      [ cv2.resize(mask,(fixedwidthImg,fixedheightImg)),
                        cv2.resize(img_vh,(fixedwidthImg,fixedheightImg)), 
                        cv2.resize(bitxor,(fixedwidthImg,fixedheightImg)),
                        cv2.resize(image,(fixedwidthImg,fixedheightImg))])

        
    
    #cv2.imwrite("scanned/Original"+str(count)+".jpg",img)
    #cv2.imwrite("scanned/Warp-Prespective"+str(count)+".jpg",imgWarpColored)
    # LABELS FOR DISPLAY
    lables = [["Original","Binary","Vertical","Horizontal"],
              ["Mask","BITXOR","BITXOR","Contours"]]

    stackedImage = my_utils.stackImages(imageArray,0.75,lables)
    cv2.imshow("PROCESAMIENTO DE TABLA",stackedImage)
    
def random_color():
    rgbl=[255,0,0]
    random.shuffle(rgbl)
    return tuple(rgbl)