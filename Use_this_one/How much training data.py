import os

'''
This is a file that is designed to run the other files in my AI
so that I can see the most efficient amount of training/test
images to use to get the highest accuracy.
'''

# import all neccessary files
path = r'D:\School Stuff\CBASE\Master_Project\Pre-Built_AI\Use_this_one'
# Import these as modules so we can call their functions later
import Move_Files_Updated
import Move_Files_Updated_Reverse

# Reset the Image library
Move_Files_Updated_Reverse.Move_Files_Updated_Reverse()

# number of images to start with in each path
num_desired = 100
accuracy_array = []

# while num_desired <= 500001:
while num_desired <= 2000:
    # Deletes previous cache - necessary for everything to work properly
    exec(open('Delete_Previous_Dataset_Cache.py').read())

    # Moves more images into Image directory and prints current amount
    Move_Files_Updated.Move_Files_Updated(num_desired)
    print('number of pictures in each folder = ', num_desired)

    # This is where the fun begins
    exec(open(path + '\\' + 'dataset.py').read())
    exec(open(path + '\\' + 'Train_CD.py').read())

    # Get accuracy percentage and add it to the array
    # accuracy_array.append(#number from accuracy percentage)

    exec(open(path + '\\' + 'Running.py').read())

    # number of images to add to each path
    num_desired += 1000

    # Just a failsafe
    if num_desired > 500000:
        break
