from __future__ import division
import math
import os
import numpy as np
import cv2
import tensorflow
from keras.models import load_model
from PIL import Image

# Variables
######################

threshold = 0.5 # Probability threshold


model = load_model('my_model_Crack.h5')

dir = r'D:\School Stuff\CBASE\Different_aI\Cracks to Test\\'
name = 'crack1'
type = '.jpg'
file = dir + name + type

# Reading an image in default mode and creating subimages
i = Image.open(file)
width, height = i.size
print(width)
print(height)
W=width
H=height

# Top Portion
frame1 = i.crop((0, 0, W/3, H/3))
frame2 = i.crop((W/3, 0, 2*W/3, H/3))
frame3= i.crop((2*W/3, 0, W, H/3))
# Middle Portion
frame4 = i.crop((0, H/3, W/3, 2*H/3))
frame5 = i.crop((W/3, H/3, 2*W/3, 2*H/3))
frame6 = i.crop((2*W/3, H/3, W, 2*H/3))
# Bottom Portion
frame7 = i.crop((0, 2*H/3, W/3, H))
frame8 = i.crop((W/3, 2*H/3, 2*W/3, H))
frame9 = i.crop((2*W/3, 2*H/3, W, H))

frames = [frame1, frame2, frame3, frame4, frame5, frame6, frame7, frame8, frame9]

# Rectangles for Top Portion
rect11 = (0, 0); rect12 = (int(W/3), int(H/3))
rect21 = (int(W/3), 0); rect22 = (int(2*W/3), int(H/3))
rect31 = (int(2*W/3), 0); rect32 = (int(W), int(H/3))
# Rectangles for Middle Portion
rect41 = (0, int(H/3)); rect42 = (int(W/3), int(2*H/3))
rect51 = (int(W/3), int(H/3)); rect52 = (int(2*W/3), int(2*H/3))
rect61 = (int(2*W/3), int(H/3)); rect62 = (int(W), int(2*H/3))
# Rectangles for Bottom Portion
rect71 = (0, int(2*H/3)); rect72 = (int(W/3), int(H))
rect81 = (int(W/3), int(2*H/3)); rect82 = (int(2*W/3), int(H))
rect91 = (int(2*W/3), int(2*H/3)); rect92 = (int(W), int(H))

rectangles = [rect11, rect12, rect21, rect22, rect31, rect32, rect41, rect42, rect51, rect52, rect61, rect62, rect71, rect72, rect81, rect82, rect91, rect92]
####3##

def preProcessing(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        img = img / 255
        return img

num = 1
d={}
for frame in frames:
    imgOriginal = frame
    img = np.asarray(imgOriginal)
    img = cv2.resize(img,(500,500))
    img = preProcessing(img)
    cv2.imshow("Processed Image",img)

    img = img.reshape(1,500,500,1)
    # Predict
    classIndex = int(model.predict_classes(img))
    #print(classIndex)
    predictions = model.predict(img) # Gets probability that it thinks object is from each given class
    #print(predictions)
    probVal = np.amax(predictions) # Gets max one
    if probVal > threshold:

        if classIndex == 0:
            d["item{0}".format(num)] = "Crack"
            d["prob{0}".format(num)] = probVal
            #print(item#, "Probability: ",probVal)
        elif classIndex == 1:
            d["item{0}".format(num)] = "No_Crack"
            d["prob{0}".format(num)] = probVal
            #print(item#, "Probability: ",probVal)
    num += 1
#print(d)

# path
#path = r'C:\Users\drbah\Desktop\EN courses\EN513\students projects\Final_project\New folder\project3\myData\crack1.jpg'

# Reading an image in default mode
image = cv2.imread(file)

# Window name in which image is displayed
window_name = 'Image'

# font
font = cv2.FONT_HERSHEY_SIMPLEX

# org
org1 = (int(W/8), int(H/6))
org2 = (int(4*W/8), int(H/6))
org3 = (int(7*W/8), int(H/6))
org4 = (int(W/8), int(3*H/6))
org5 = (int(4*W/8), int(3*H/6))
org6 = (int(7*W/8), int(3*H/6))
org7 = (int(W/8), int(5*H/6))
org8 = (int(4*W/8), int(5*H/6))
org9 = (int(7*W/8), int(5*H/6))

# fontScale
fontScale = .35

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 1 px
thickness = 1
# Using cv2.putText() method
image = cv2.putText(image, d['item1'] + "  " + str(d['prob1']), org1, font, fontScale, color, thickness, cv2.LINE_AA)
image = cv2.putText(image, d['item2'] + "  " + str(d['prob2']), org2, font, fontScale, color, thickness, cv2.LINE_AA)
image = cv2.putText(image, d['item3'] + "  " + str(d['prob3']), org3, font, fontScale, color, thickness, cv2.LINE_AA)
image = cv2.putText(image, d['item4'] + "  " + str(d['prob4']), org4, font, fontScale, color, thickness, cv2.LINE_AA)
image = cv2.putText(image, d['item5'] + "  " + str(d['prob5']), org5, font, fontScale, color, thickness, cv2.LINE_AA)
image = cv2.putText(image, d['item6'] + "  " + str(d['prob6']), org6, font, fontScale, color, thickness, cv2.LINE_AA)
image = cv2.putText(image, d['item7'] + "  " + str(d['prob7']), org7, font, fontScale, color, thickness, cv2.LINE_AA)
image = cv2.putText(image, d['item8'] + "  " + str(d['prob8']), org8, font, fontScale, color, thickness, cv2.LINE_AA)
image = cv2.putText(image, d['item9'] + "  " + str(d['prob9']), org9, font, fontScale, color, thickness, cv2.LINE_AA)

image = cv2.rectangle(image, rect11, rect12, color, thickness)
image = cv2.rectangle(image, rect21, rect22, color, thickness)
image = cv2.rectangle(image, rect31, rect32, color, thickness)
image = cv2.rectangle(image, rect41, rect42, color, thickness)
image = cv2.rectangle(image, rect51, rect52, color, thickness)
image = cv2.rectangle(image, rect61, rect62, color, thickness)
image = cv2.rectangle(image, rect71, rect72, color, thickness)
image = cv2.rectangle(image, rect81, rect82, color, thickness)
image = cv2.rectangle(image, rect91, rect92, color, thickness)

new_dir = r'D:\School Stuff\CBASE\Different_aI\Tested_Data\500_500\\'
cv2.imwrite(
        new_dir + name + 'location' + type,
        image)
