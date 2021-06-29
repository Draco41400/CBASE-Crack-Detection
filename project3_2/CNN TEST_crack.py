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
name = 'crack338'
type = '.jpg'
file = dir + name + type

# Reading an image in default mode and creating subimages
i = Image.open(file)
width, height = i.size
print(width)
print(height)
W=width
H=height

# Rectangles
# First line
rect11 = (0,0);           rect12 = (int(W/5), int(H/5))
rect21 = (int(W/5), 0);   rect22 = (int(2*W/5), int(H/5))
rect31 = (int(2*W/5), 0); rect32 = (int(3*W/5), int(H/5))
rect41 = (int(3*W/5), 0); rect42 = (int(4*W/5), int(H/5))
rect51 = (int(4*W/5), 0); rect52 = (int(W), int(H/5))
# Second Line
rect61 = (0, int(H/5));           rect62 = (int(W/5), int(2*H/5))
rect71 = (int(W/5), int(H/5));    rect72 = (int(2*W/5), int(2*H/5))
rect81 = (int(2*W/5), int(H/5));  rect82 = (int(3*W/5), int(2*H/5))
rect91 = (int(3*W/5), int(H/5));  rect92 = (int(4*W/5), int(2*H/5))
rect101 = (int(4*W/5), int(H/5)); rect102 = (int(W), int(2*H/5))
# Third Line
rect111 = (0, int(2*H/5));          rect112 = (int(W/5), int(3*H/5))
rect121 = (int(W/5), int(2*H/5));   rect122 = (int(2*W/5), int(3*H/5))
rect131 = (int(2*W/5), int(2*H/5)); rect132 = (int(3*W/5), int(3*H/5))
rect141 = (int(3*W/5), int(2*H/5)); rect142 = (int(4*W/5), int(3*H/5))
rect151 = (int(4*W/5), int(2*H/5)); rect152 = (int(W), int(3*H/5))
# Fourth Line
rect161 = (0, int(3*H/5));          rect162 = (int(W/5), int(4*H/5))
rect171 = (int(W/5), int(3*H/5));   rect172 = (int(2*W/5), int(4*H/5))
rect181 = (int(2*W/5), int(3*H/5)); rect182 = (int(3*W/5), int(4*H/5))
rect191 = (int(3*W/5), int(3*H/5)); rect192 = (int(4*W/5), int(4*H/5))
rect201 = (int(4*W/5), int(3*H/5)); rect202 = (int(W), int(4*H/5))
# Fifth Line
rect211 = (0, int(4*H/5));          rect212 = (int(W/5), int(H))
rect221 = (int(W/5), int(4*H/5));   rect222 = (int(2*W/5), int(H))
rect231 = (int(2*W/5), int(4*H/5)); rect232 = (int(3*W/5), int(H))
rect241 = (int(3*W/5), int(4*H/5)); rect242 = (int(4*W/5), int(H))
rect251 = (int(4*W/5), int(4*H/5)); rect252 = (int(W), int(H))

# Frames
# First line
frame1 = i.crop((rect11[0], rect11[1], rect12[0], rect12[1]))
frame2 = i.crop((rect21[0], rect21[1], rect22[0], rect22[1]))
frame3 = i.crop((rect31[0], rect31[1], rect32[0], rect32[1]))
frame4 = i.crop((rect41[0], rect41[1], rect42[0], rect42[1]))
frame5 = i.crop((rect51[0], rect51[1], rect52[0], rect52[1]))
# Second Line
frame6 = i.crop((rect61[0], rect61[1], rect62[0], rect62[1]))
frame7 = i.crop((rect71[0], rect71[1], rect72[0], rect72[1]))
frame8 = i.crop((rect81[0], rect81[1], rect82[0], rect82[1]))
frame9 = i.crop((rect91[0], rect91[1], rect92[0], rect92[1]))
frame10 = i.crop((rect101[0], rect101[1], rect102[0], rect102[1]))
# Third Line
frame11 = i.crop((rect111[0], rect111[1], rect112[0], rect112[1]))
frame12 = i.crop((rect121[0], rect121[1], rect122[0], rect122[1]))
frame13 = i.crop((rect131[0], rect131[1], rect132[0], rect132[1]))
frame14 = i.crop((rect141[0], rect141[1], rect142[0], rect142[1]))
frame15 = i.crop((rect151[0], rect151[1], rect152[0], rect152[1]))
# Fourth Line
frame16 = i.crop((rect161[0], rect161[1], rect162[0], rect162[1]))
frame17 = i.crop((rect171[0], rect171[1], rect172[0], rect172[1]))
frame18 = i.crop((rect181[0], rect181[1], rect182[0], rect182[1]))
frame19 = i.crop((rect191[0], rect191[1], rect192[0], rect192[1]))
frame20 = i.crop((rect201[0], rect201[1], rect202[0], rect202[1]))
# Fifth Line
frame21 = i.crop((rect211[0], rect211[1], rect212[0], rect212[1]))
frame22 = i.crop((rect221[0], rect221[1], rect222[0], rect222[1]))
frame23 = i.crop((rect231[0], rect231[1], rect232[0], rect232[1]))
frame24 = i.crop((rect241[0], rect241[1], rect242[0], rect242[1]))
frame25 = i.crop((rect251[0], rect251[1], rect252[0], rect252[1]))

frames = [frame1, frame2, frame3, frame4, frame5, frame6, frame7, frame8, frame9, frame10, frame11, frame12, frame13, frame14, frame15, frame16, frame17, frame18, frame19, frame20, frame21, frame22, frame23, frame24, frame25]
'''
# Rectangles for Top Portion
rect11 = (0, 0);          rect12 = (int(W/3), int(H/3))
rect21 = (int(W/3), 0);   rect22 = (int(2*W/3), int(H/3))
rect31 = (int(2*W/3), 0); rect32 = (int(W), int(H/3))
# Rectangles for Middle Portion
rect41 = (0, int(H/3));          rect42 = (int(W/3), int(2*H/3))
rect51 = (int(W/3), int(H/3));   rect52 = (int(2*W/3), int(2*H/3))
rect61 = (int(2*W/3), int(H/3)); rect62 = (int(W), int(2*H/3))
# Rectangles for Bottom Portion
rect71 = (0, int(2*H/3));          rect72 = (int(W/3), int(H))
rect81 = (int(W/3), int(2*H/3));   rect82 = (int(2*W/3), int(H))
rect91 = (int(2*W/3), int(2*H/3)); rect92 = (int(W), int(H))

# Top Portion
frame1 = i.crop((rect11[0], rect11[1], rect12[0], rect12[1]))
frame2 = i.crop((rect21[0], rect21[1], rect22[0], rect22[1]))
frame3 = i.crop((rect31[0], rect31[1], rect32[0], rect32[1]))
# Middle Portion
frame4 = i.crop((rect41[0], rect41[1], rect42[0], rect42[1]))
frame5 = i.crop((rect51[0], rect51[1], rect52[0], rect52[1]))
frame6 = i.crop((rect61[0], rect61[1], rect62[0], rect62[1]))
# Bottom Portion
frame7 = i.crop((rect71[0], rect71[1], rect72[0], rect72[1]))
frame8 = i.crop((rect81[0], rect81[1], rect82[0], rect82[1]))
frame9 = i.crop((rect91[0], rect91[1], rect92[0], rect92[1]))

#frames = [frame1, frame2, frame3, frame4, frame5, frame6, frame7, frame8, frame9]

num = 1
frames = {}
while num <= 9:
    frames["frame{0}".format(num)] = i.crop(("rect{0}1".format(num)[0], "rect{0}1".format(num)[1], "rect{0}2".format(num)[0], "rect{0}2".format(num)[1]))
    #frames.append("frame{0}".format(num))
    num += 1
print(frames)
'''
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
    img = cv2.resize(img,(50,50))
    img = preProcessing(img)
    cv2.imshow("Processed Image",img)

    img = img.reshape(1,50,50,1)
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
print(d)

# path
#path = r'C:\Users\drbah\Desktop\EN courses\EN513\students projects\Final_project\New folder\project3\myData\crack1.jpg'

# Reading an image in default mode
image = cv2.imread(file)

# Window name in which image is displayed
window_name = 'Image'

# font
font = cv2.FONT_HERSHEY_SIMPLEX
'''
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
'''
org1 = (np.array(rect11) + np.array(rect12)) / 2
org1 = tuple((int(org1[0]), int(org1[1])))
org2 = (np.array(rect21) + np.array(rect22)) / 2
org2 = tuple((int(org2[0]), int(org2[1])))
org3 = (np.array(rect31) + np.array(rect32)) / 2
org3 = tuple((int(org3[0]), int(org3[1])))
org4 = (np.array(rect41) + np.array(rect42)) / 2
org4 = tuple((int(org4[0]), int(org4[1])))
org5 = (np.array(rect51) + np.array(rect52)) / 2
org5 = tuple((int(org5[0]), int(org5[1])))
org6 = (np.array(rect61) + np.array(rect62)) / 2
org6 = tuple((int(org6[0]), int(org6[1])))
org7 = (np.array(rect71) + np.array(rect72)) / 2
org7 = tuple((int(org7[0]), int(org7[1])))
org8 = (np.array(rect81) + np.array(rect82)) / 2
org8 = tuple((int(org8[0]), int(org8[1])))
org9 = (np.array(rect91) + np.array(rect92)) / 2
org9 = tuple((int(org9[0]), int(org9[1])))
org10 = (np.array(rect101) + np.array(rect102)) / 2
org10 = tuple((int(org10[0]), int(org10[1])))
org11 = (np.array(rect111) + np.array(rect112)) / 2
org11 = tuple((int(org11[0]), int(org11[1])))
org12 = (np.array(rect121) + np.array(rect122)) / 2
org12 = tuple((int(org12[0]), int(org12[1])))
org13 = (np.array(rect131) + np.array(rect132)) / 2
org13 = tuple((int(org13[0]), int(org13[1])))
org14 = (np.array(rect141) + np.array(rect142)) / 2
org14 = tuple((int(org14[0]), int(org14[1])))
org15 = (np.array(rect151) + np.array(rect152)) / 2
org15 = tuple((int(org15[0]), int(org15[1])))
org16 = (np.array(rect161) + np.array(rect162)) / 2
org16 = tuple((int(org16[0]), int(org16[1])))
org17 = (np.array(rect171) + np.array(rect172)) / 2
org17 = tuple((int(org17[0]), int(org17[1])))
org18 = (np.array(rect181) + np.array(rect182)) / 2
org18 = tuple((int(org18[0]), int(org18[1])))
org19 = (np.array(rect191) + np.array(rect192)) / 2
org19 = tuple((int(org19[0]), int(org19[1])))
org20 = (np.array(rect201) + np.array(rect202)) / 2
org20 = tuple((int(org20[0]), int(org20[1])))
org21 = (np.array(rect211) + np.array(rect212)) / 2
org21 = tuple((int(org21[0]), int(org21[1])))
org22 = (np.array(rect221) + np.array(rect222)) / 2
org22 = tuple((int(org22[0]), int(org22[1])))
org23 = (np.array(rect231) + np.array(rect232)) / 2
org23 = tuple((int(org23[0]), int(org23[1])))
org24 = (np.array(rect241) + np.array(rect242)) / 2
org24 = tuple((int(org24[0]), int(org24[1])))
org25 = (np.array(rect251) + np.array(rect252)) / 2
org25 = tuple((int(org25[0]), int(org25[1])))
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
image = cv2.putText(image, d['item10'] + "  " + str(d['prob10']), org10, font, fontScale, color, thickness, cv2.LINE_AA)
image = cv2.putText(image, d['item11'] + "  " + str(d['prob11']), org11, font, fontScale, color, thickness, cv2.LINE_AA)
image = cv2.putText(image, d['item12'] + "  " + str(d['prob12']), org12, font, fontScale, color, thickness, cv2.LINE_AA)
image = cv2.putText(image, d['item13'] + "  " + str(d['prob13']), org13, font, fontScale, color, thickness, cv2.LINE_AA)
image = cv2.putText(image, d['item14'] + "  " + str(d['prob14']), org14, font, fontScale, color, thickness, cv2.LINE_AA)
image = cv2.putText(image, d['item15'] + "  " + str(d['prob15']), org15, font, fontScale, color, thickness, cv2.LINE_AA)
image = cv2.putText(image, d['item16'] + "  " + str(d['prob16']), org16, font, fontScale, color, thickness, cv2.LINE_AA)
image = cv2.putText(image, d['item17'] + "  " + str(d['prob17']), org17, font, fontScale, color, thickness, cv2.LINE_AA)
image = cv2.putText(image, d['item18'] + "  " + str(d['prob18']), org18, font, fontScale, color, thickness, cv2.LINE_AA)
image = cv2.putText(image, d['item19'] + "  " + str(d['prob19']), org19, font, fontScale, color, thickness, cv2.LINE_AA)
image = cv2.putText(image, d['item20'] + "  " + str(d['prob20']), org20, font, fontScale, color, thickness, cv2.LINE_AA)
image = cv2.putText(image, d['item21'] + "  " + str(d['prob21']), org21, font, fontScale, color, thickness, cv2.LINE_AA)
image = cv2.putText(image, d['item22'] + "  " + str(d['prob22']), org22, font, fontScale, color, thickness, cv2.LINE_AA)
image = cv2.putText(image, d['item23'] + "  " + str(d['prob23']), org23, font, fontScale, color, thickness, cv2.LINE_AA)
image = cv2.putText(image, d['item24'] + "  " + str(d['prob24']), org24, font, fontScale, color, thickness, cv2.LINE_AA)
image = cv2.putText(image, d['item25'] + "  " + str(d['prob25']), org25, font, fontScale, color, thickness, cv2.LINE_AA)

image = cv2.rectangle(image, rect11, rect12, color, thickness)
image = cv2.rectangle(image, rect21, rect22, color, thickness)
image = cv2.rectangle(image, rect31, rect32, color, thickness)
image = cv2.rectangle(image, rect41, rect42, color, thickness)
image = cv2.rectangle(image, rect51, rect52, color, thickness)
image = cv2.rectangle(image, rect61, rect62, color, thickness)
image = cv2.rectangle(image, rect71, rect72, color, thickness)
image = cv2.rectangle(image, rect81, rect82, color, thickness)
image = cv2.rectangle(image, rect91, rect92, color, thickness)
image = cv2.rectangle(image, rect101, rect102, color, thickness)
image = cv2.rectangle(image, rect111, rect112, color, thickness)
image = cv2.rectangle(image, rect121, rect122, color, thickness)
image = cv2.rectangle(image, rect131, rect132, color, thickness)
image = cv2.rectangle(image, rect141, rect142, color, thickness)
image = cv2.rectangle(image, rect151, rect152, color, thickness)
image = cv2.rectangle(image, rect161, rect162, color, thickness)
image = cv2.rectangle(image, rect171, rect172, color, thickness)
image = cv2.rectangle(image, rect181, rect182, color, thickness)
image = cv2.rectangle(image, rect191, rect192, color, thickness)
image = cv2.rectangle(image, rect201, rect202, color, thickness)
image = cv2.rectangle(image, rect211, rect212, color, thickness)
image = cv2.rectangle(image, rect221, rect222, color, thickness)
image = cv2.rectangle(image, rect231, rect232, color, thickness)
image = cv2.rectangle(image, rect241, rect242, color, thickness)
image = cv2.rectangle(image, rect251, rect252, color, thickness)

new_dir = r'D:\School Stuff\CBASE\Different_aI\Tested_Data\50_50\\'
cv2.imwrite(
        new_dir + name + 'location' + type,
        image)
