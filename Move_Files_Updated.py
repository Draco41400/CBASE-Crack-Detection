import os
import shutil
from PIL import Image

def Move_Files_Updated(num_desire):

    # Don't worry about this one
    # im = Image.open(r'D:\Pictures\Shadow Realm Jimbo.jpg')


    # Labeling all paths
    source_crack = r'D:\School Stuff\CBASE\Master_Project\images_2mil\crack'
    source_no_crack = r'D:\School Stuff\CBASE\Master_Project\images_2mil\no_crack'

    destination_crack = r'D:\School Stuff\CBASE\Master_Project\temp_images\crack'
    destination_crack_test = r'D:\School Stuff\CBASE\Master_Project\temp_images\crack\test'
    destination_no_crack = r'D:\School Stuff\CBASE\Master_Project\temp_images\no_crack'
    destination_no_crack_test = r'D:\School Stuff\CBASE\Master_Project\temp_images\no_crack\test'

    destination_final_crack = [destination_crack, destination_crack_test]
    destination_final_no_crack =[destination_no_crack, destination_no_crack_test]

    files_crack = os.listdir(source_crack)
    files_no_crack = os.listdir(source_no_crack)

    # Crack Moving
    for path in destination_final_crack:
        num_files = len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))])
        print(num_files)
        for file in files_crack:
            new_path = shutil.move(f"{source_crack}\{file}", path)
            num_files += 1
            if num_files >= num_desire:
                files_crack = os.listdir(source_crack)
                print (num_files)
                print("That's enough images Jimbo")
                break

    # Non Crack Moving
    for path in destination_final_no_crack:
        num_files = len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))])
        print(num_files)
        for file in files_no_crack:
            new_path = shutil.move(f"{source_no_crack}\{file}", path)
            num_files += 1
            if num_files >= num_desire:
                files_no_crack = os.listdir(source_no_crack)
                print (num_files)
                print("That's enough images Jimbo")
                break
    # im.show()
