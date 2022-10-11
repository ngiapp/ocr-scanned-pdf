# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 17:14:50 2022

@author: NGIAPKOH
"""

import cv2
import numpy as np
from pdf2image import convert_from_path

# Read image 
doc = convert_from_path('C:/Micron/Archiving/Soonest/SEC138520407-HAWB.pdf', 500, poppler_path = r'C:\Poppler\poppler-0.68.0_x86\poppler-0.68.0\bin')
doc[0].save('C:/Micron/Archiving/Soonest/Test.jpg')

img = cv2.imread('C:/Micron/Archiving/Soonest/Test.jpg', cv2.IMREAD_COLOR) # road.png is the filename
# Convert the image to gray-scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Find the edges in the image using canny detector
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
thresh = cv2.dilate(thresh, None, iterations=2)
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))
detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=10)
cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts_hori = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts_hori:
    cv2.drawContours(img, [c], -1, (0,0,0), 3)


vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,20))
detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=10)
cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts_vert = cnts[0] if len(cnts) == 2 else cnts[1]

for c in cnts_vert:
    cv2.drawContours(img, [c], -1, (0,0,0), 3)

line_coords_list = []

# for cnt in cnts_vert:
#     x,y,w,h = cv2.boundingRect(cnt)
#     line_coords_list.append((x,y,w,h))
#     end_y_coordinate = y+h
#     end_x_coordinate = x+w
#     temp_img = img[y : end_y_coordinate , x : end_x_coordinate]
#     temp_name = 'result' + "_"+str(y)+"_"+str(end_y_coordinate)+ "_" + str(x) + "_" + str(end_x_coordinate) + ".png"
#     cv2.imwrite('C:/Micron/Archiving/Soonest/'+temp_name, temp_img)
    
    
cv2.imwrite('C:/Micron/Archiving/Soonest/cnts.png',img)

image = cv2.imread('C:/Micron/Archiving/Soonest/cnts.png')
image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
lines = cv2.HoughLinesP(image,rho=1,theta=np.pi/180,threshold=10,minLineLength=200, maxLineGap=5)
a,b,c = lines.shape

line_coords_list = []

_, blackAndWhite = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)
contours,h = cv2.findContours(blackAndWhite,cv2.RETR_LIST ,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    line_coords_list.append((x,y,w,h))
    temp_img = image[y : y+h , x : x+w]
    temp_name = 'result' + "_"+str(y )+"_"+str(y+h)+ "_" + str(x) + "_" + str(x+w) + ".png"
    cv2.imwrite('C:/Micron/Archiving/Soonest/Contours/'+temp_name, temp_img)
# for c in cnts:
#     cv2.drawContours(image, [c], -1, (36,255,12), 3)
# # Detect points that form a line
# # =============================================================================
# # 
# # =============================================================================
# for i in range(1,30):
#     try:
#         lines = cv2.HoughLinesP(thresh,rho=1,theta=np.pi/180,threshold=10,minLineLength=200, maxLineGap=i*1)
#         # Draw lines on the image
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
#         # Show result
#         # cv2.imshow("Result Image", img)
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()
        

#         cv2.imwrite('C:/Micron/Archiving/Soonest/hough-%s.jpg' % i,img)
#     except:
#         continue