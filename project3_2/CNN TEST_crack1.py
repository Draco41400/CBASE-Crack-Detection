from __future__ import division
import math
import os
import numpy as np
import cv2
from keras.models import load_model
import Image


# Variables
######################

threshold = 0.95


model = load_model('my_model_Crack.h5')

path = r'C:\Users\drbah\Desktop\EN courses\EN513\students projects\Final_project\New folder\project3\myData\crack2.jpg'

# Reading an image in default mode and creating subimages
i = Image.open(path)
width, height = i.size
print(width)
print(height)
L=0
T=0
R=width
B=height

frame1 = i.crop(((L, T, R, B/3)))
#frame1.save('crack_frame2.jpg')
frame2 = i.crop(((L, B/3, R, 2*B/3)))
#frame2.save('crack_frame3.jpg')
frame3= i.crop(((L, 2*B/3, R, B )))
#frame3.save('crack_frame4.jpg')
####3##

def preProcessing(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        img = img / 255
        return img



imgOriginal= frame1
img = np.asarray(imgOriginal)
img = cv2.resize(img,(32,32))
img = preProcessing(img)
cv2.imshow("Processed Image",img)

img = img.reshape(1,32,32,1)
# Predict
classIndex = int(model.predict_classes(img))
#print(classIndex)
predictions = model.predict(img) # Gets probability that it thinks object is from each given class
#print(predictions)
probVal = np.amax(predictions) # Gets max one
if probVal > threshold:

    if classIndex == 0:
        item1 = "Crack"
        prob1=probVal
        #print(item1, "Probability: ",probVal)
    elif classIndex == 1:
        item1 = "No_Crack"
        prob1=probVal
        #print(item1, "Probability: ",probVal)


imgOriginal= frame2
img = np.asarray(imgOriginal)
img = cv2.resize(img,(32,32))
img = preProcessing(img)
cv2.imshow("Processed Image",img)

img = img.reshape(1,32,32,1)
# Predict
classIndex = int(model.predict_classes(img))
#print(classIndex)
predictions = model.predict(img) # Gets probability that it thinks object is from each given class
#print(predictions)
probVal = np.amax(predictions) # Gets max one
if probVal > threshold:

    if classIndex == 0:
        item2 = "Crack"
        prob2=probVal
        #print(item2, "Probability: ",probVal)
    elif classIndex == 1:
        item2 = "No_Crack"
        prob2=probVal
       # print(item2, "Probability: ",probVal)




imgOriginal= frame3
img = np.asarray(imgOriginal)
img = cv2.resize(img,(32,32))
img = preProcessing(img)
cv2.imshow("Processed Image",img)

img = img.reshape(1,32,32,1)
# Predict
classIndex = int(model.predict_classes(img))
#print(classIndex)
predictions = model.predict(img) # Gets probability that it thinks object is from each given class
#print(predictions)
probVal = np.amax(predictions) # Gets max one
if probVal > threshold:
   if classIndex == 0:
        item3 = "Crack"
        prob3=probVal
       # print(item3, "Probability: ",probVal)
   elif classIndex == 1:
        item3 = "No_Crack"
        prob3=probVal
       # print(item3, "Probability: ",probVal)



# path
#path = r'C:\Users\drbah\Desktop\EN courses\EN513\students projects\Final_project\New folder\project3\myData\crack1.jpg'

# Reading an image in default mode
image = cv2.imread(path)

# Window name in which image is displayed
window_name = 'Image'

# font
font = cv2.FONT_HERSHEY_SIMPLEX

# org
org1 = (int(R/8), int(B/6))
org2 = (int(R/8), int(3*B/6))
org3 = (int(R/8), int(5*B/6))
# fontScale
fontScale = .35

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 1 px
thickness = 1

# Using cv2.putText() method
image = cv2.putText(image, item1+ "  "+str(prob1), org1, font, fontScale, color, thickness, cv2.LINE_AA)
image = cv2.putText(image, item2+ "  "+str(prob2), org2, font, fontScale, color, thickness, cv2.LINE_AA)
image = cv2.putText(image, item3+ "  "+str(prob3), org3, font, fontScale, color, thickness, cv2.LINE_AA)



cv2.imwrite(
        r'C:\Users\drbah\Desktop\EN courses\EN513\students projects\Final_project\New folder\project3\Cracklocation1.jpg',
        image)

