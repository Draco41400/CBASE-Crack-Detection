import os
import shutil
from PIL import Image

def Move_Files_Updated_Reverse():

    # Don't worry about this one
    im = Image.open(r'D:\Pictures\Shadow Realm Jimbo.jpg')
    im.show()


    # Labeling all paths
    source_crack = r'D:\School Stuff\CBASE\Master_Project\temp_images\crack'
    source_crack_test = r'D:\School Stuff\CBASE\Master_Project\temp_images\crack\test'
    source_no_crack = r'D:\School Stuff\CBASE\Master_Project\temp_images\no_crack'
    source_no_crack_test = r'D:\School Stuff\CBASE\Master_Project\temp_images\no_crack\test'

    destination_crack = r'D:\School Stuff\CBASE\Master_Project\images_2mil\crack'
    destination_no_crack = r'D:\School Stuff\CBASE\Master_Project\images_2mil\no_crack'

    files_crack = os.listdir(source_crack)
    files_crack_test = os.listdir(source_crack_test)
    files_no_crack = os.listdir(source_no_crack)
    files_no_crack_test = os.listdir(source_no_crack_test)

    # Crack Moving
    num_files = len([f for f in os.listdir(source_crack)if os.path.isfile(os.path.join(source_crack, f))])
    print(num_files)
    for file in files_crack:
        if file.endswith('.jpg'):
            new_path = shutil.move(f"{source_crack}/{file}", destination_crack)
            num_files -= 1
            print (num_files)
            if num_files <= 1:
                break
        if file.endswith('.png'):
            new_path = shutil.move(f"{source_crack}/{file}", destination_crack)
            num_files -= 1
            if num_files <= 1:
                break

    num_files = len([f for f in os.listdir(source_crack_test)if os.path.isfile(os.path.join(source_crack_test, f))])
    print(num_files)
    for file in files_crack_test:
        if file.endswith('.jpg'):
            new_path = shutil.move(f"{source_crack_test}/{file}", destination_crack)
            num_files -= 1
            print (num_files)
            if num_files <= 0:
                break
        if file.endswith('.png'):
            new_path = shutil.move(f"{source_crack_test}/{file}", destination_crack)
            num_files -= 1
            print (num_files)
            if num_files <= 0:
                break

    # Non Crack Moving
    num_files = len([f for f in os.listdir(source_no_crack)if os.path.isfile(os.path.join(source_no_crack, f))])
    print(num_files)
    for file in files_no_crack:
        if file.endswith('.jpg'):
            new_path = shutil.move(f"{source_no_crack}/{file}", destination_no_crack)
            num_files -= 1
            print (num_files)
            if num_files <= 0:
                break
        if file.endswith('.png'):
            new_path = shutil.move(f"{source_no_crack}/{file}", destination_no_crack)
            num_files -= 1
            print (num_files)
            if num_files <= 0:
                break

    num_files = len([f for f in os.listdir(source_no_crack_test)if os.path.isfile(os.path.join(source_no_crack_test, f))])
    print(num_files)
    for file in files_no_crack_test:
        if file.endswith('.jpg'):
            new_path = shutil.move(f"{source_no_crack_test}/{file}", destination_no_crack)
            num_files -= 1
            print (num_files)
            if num_files <= 0:
                break
        if file.endswith('.png'):
            new_path = shutil.move(f"{source_no_crack_test}/{file}", destination_no_crack)
            num_files -= 1
            print (num_files)
            if num_files <= 0:
                break
