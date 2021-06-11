import numpy as np
import tensorflow as tf
print(tf.__version__)
import dill
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.figure
from keras.preprocessing.image import  ImageDataGenerator
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
#import pickle

### Variables
#######################
main_path = r'D:\School Stuff\CBASE\Separate'
path = main_path + '\\' + 'myData'
pathLabels = 'labels.csv'
testRatio = 0.1
valRatio = 0.1
imageDimensions = (50,50,3)

#batchSizeVal = 50
#epochsVal = 6
#stepsPerEpochVal = 2000
#######################

### Getting Data
images = []
classNo = []
myList = os.listdir(path)
print("Total No of Classes Detected:",len(myList))
noOfClasses = len(myList)
print("Importing Classes....")
for x in range (0,noOfClasses):
    myPicList = os.listdir(path+"/"+str(x))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(x)+"/"+y)
        curImg = cv2.resize(curImg,(imageDimensions[0],imageDimensions[1]))
        images.append(curImg)
        classNo.append(x)
    print(x,end = " ")
print(" ")
print(len(images))

images = np.array(images) ### turn into array
classNo = np.array(classNo)


#im = list(images.getdata())
#flattened = im.flatten()
#print(flattened.shape)

#print (images.shape)
#print(classNo.shape)

### Spliting the Data
X_train,X_test,y_train,y_test = train_test_split(images,classNo,test_size=testRatio)
X_train,X_Validation,y_train,y_Validation = train_test_split(X_train,y_train,test_size=valRatio)

print(X_train.shape)
print(X_test.shape)
print(X_Validation.shape)
######Bahaa


batchSizeVal =65
epochsVal = 20
#stepsPerEpochVal = 1000

stepsPerEpochVal = len(X_train)//batchSizeVal
#validation_steps = len(X_test)//batch_size # if you have test data
#####
###find exactly where each images come from

numOfSamples = []
for x in range(0,noOfClasses):
    print(len(np.where(y_train==x)[0]))
    numOfSamples.append(len(np.where(y_train==x)[0]))
print(numOfSamples)

### Plot equal data split
plt.figure(figsize = (10,5))
plt.bar(range(0,noOfClasses),numOfSamples)
plt.title("No of Train Images for Each Class")
plt.xlabel("Class ID")
plt.xlabel("Number of Images")
plt.show()

### Pre Processing for better results when training
def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #Gray scale
    img = cv2.equalizeHist(img)
    img = img/255
    return img

#img = preProcessing(X_train[180])
#img = cv2.resize(img,(300,300))
#cv2.imshow("PreProcessed", img)
#cv2.waitKey(0)

# Apply pre processing to train, test and validation images
X_train = np.array(list(map(preProcessing,X_train)))
X_test = np.array(list(map(preProcessing,X_test)))
X_Validation = np.array(list(map(preProcessing,X_Validation)))

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2],1) # add depth of 1
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2],1)
X_Validation = X_Validation.reshape(X_Validation.shape[0], X_Validation.shape[1], X_Validation.shape[2],1)

print(X_train.shape)
print(X_test.shape)
print(X_Validation.shape)

### Generate more data for training by altering the current images
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)

dataGen.fit(X_train)

y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_Validation = to_categorical(y_Validation, noOfClasses)

### Build CNN model
def myModel():  # Based on the LeNet CNN model
    noOfFilters = 65 # number of conv filters learned
    sizeOfFilter1 = (5,5)
    sizeOfFilter2 = (3,3)
    sizeOfPool = (2,2)
    noOfNode = 500

    model = Sequential() # initializing the model
    # define first set of conv=>Activation=>pool layers
    model.add((Conv2D(noOfFilters,sizeOfFilter1,input_shape = (imageDimensions[0],
                                                               imageDimensions[1],
                                                            1), activation ='relu' ))) #rectified linear activation function or ReLU for short is a piecewise linear function that will output the input directly if it is positive, otherwise, it will output zero
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size = sizeOfPool))
    # define second set of conv=>Activation=>pool layers
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5)) # reduce or fitting and making it more generic

# Define the first Fully connected layer, making every node in previous layers connecting to every node to next layer
    model.add(Flatten())
    model.add(Dense(noOfNode,activation='relu'))
    model.add(Dropout(0.5)) # A Simple Way to Prevent Neural Networks from Overfitting
    model.add(Dense(noOfClasses, activation = 'softmax'))#defining another dense class, but accepts the number of classes as a variable
    # softmax activation returns a list of probabilities, one for each class
    model.compile(Adam(lr=0.001), loss = 'categorical_crossentropy', metrics=['accuracy']) # Using adam algorithm for optimization of the data
    return model

model = myModel()

print(model.summary())

# Actually run the training
history = model.fit_generator(dataGen.flow(X_train,y_train,
                                batch_size=batchSizeVal),
                                steps_per_epoch=stepsPerEpochVal,
                                epochs=epochsVal,
                                validation_data=(X_Validation,y_Validation),
                                shuffle=1)
#plot loss
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')

# plot accuracy
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

# get accuracy and test score values from training model onto our test values
score = model.evaluate(X_test,y_test, verbose=0)

print('Test Score = ', score[0]) # Score is the evaluation of the loss function for a given input.
print('Test Accuracy = ',score[1])
######bahaa
from keras.models import load_model

model.save('my_model_Crack.h5')  # creates a HDF5 file 'my_model.h5'
#del model  # deletes the existing model

######


# save data so it can be used in testing
#pickle_out = open("model_trained.p","wb")
#pickle.dump(model, pickle_out)
#pickle_out.close()
