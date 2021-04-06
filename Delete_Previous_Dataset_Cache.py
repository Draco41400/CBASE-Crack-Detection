import os
import fnmatch

python_dir = r'C:\Users\Zachary\AppData\Local\Programs\Python\Python37'
pycache_dir = r'D:\School Stuff\CBASE\Master_Project\Pre-Built_AI\Use_this_one\__pycache__'
checkpoint_dir = r'D:\School Stuff\CBASE\Master_Project\Pre-Built_AI\Use_this_one\Checkpoint'

python_files = os.listdir(python_dir)
pycache_files = os.listdir(pycache_dir)
checkpoint_files = os.listdir(checkpoint_dir)

for file in python_files:
    if fnmatch.fnmatch(file, '*my_dataset_cache*'):
        os.remove(file)

for file in pycache_files:
    x = pycache_dir + '\\' + file
    os.remove(x)

for file in checkpoint_files:
    x = checkpoint_dir + '\\' + file
    os.remove(x)
