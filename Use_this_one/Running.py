import numpy as np
import tensorflow.compat.v1 as tf
import os as os
from dataset import cache
from Train_CD import Model
import cv2,sys
import argparse
from pathlib import Path
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def break_image(test_image, size):

    # test_image_reshape = np.asarray(test_image)  May not be needed
    h,w= np.shape(test_image)[0],np.shape(test_image)[1]
    broken_image = []
    h_no = h//size
    w_no = w//size
    h=h_no*size
    w=w_no*size
    for i in range(0,h_no):
        for j in range(0,w_no):
            split = test_image[size*i:size*(i+1),size*j:size*(j+1),:]
            broken_image.append(split);

    return broken_image,h,w,h_no,w_no

class Dataset_test:
    def __init__(self, in_dir, exts='.png'):
        # Extend the input directory to the full path.
        in_dir = os.path.abspath(r'D:\School Stuff\CBASE\Master_Project\temp_images')

        # Input directory.
        self.in_dir = in_dir

        model=Model(in_dir)
        # Convert all file-extensions to lower-case.
        self.exts = tuple(ext.lower() for ext in exts)

        # Filenames for all the files in the test-set
        self.filenames = []

        # Class-number for each file in the test-set.
        self.class_numbers_test = []

        # Total number of classes in the data-set.
        self.num_classes = model.num_classes

        # If it is a directory.
        if os.path.isdir(in_dir):

            # Get all the valid filenames in the dir
            self.filenames = self._get_filenames_and_paths(in_dir)

        else:
            print("Invalid Directory")
        self.images = self.load_images(self.filenames)

    def _get_filenames_and_paths(self, dir):
        """
        Create and return a list of filenames with matching extensions in the given directory.
        :param dir:
            Directory to scan for files. Sub-dirs are not scanned.
        :return:
            List of filenames. Only filenames. Does not include the directory.
        """

        # Initialize empty list.
        filenames = []

        # If the directory exists.
        if os.path.exists(dir):
            # Get all the filenames with matching extensions.
            for filename in os.listdir(dir):
                if filename.lower().endswith(self.exts):
                    path = os.path.join(self.in_dir, filename)
                    filenames.append(os.path.abspath(path))

        return filenames


    def load_images(self,image_paths):
        # Load the images from disk.
        images = [cv2.imread(path) for path in image_paths]

        # Convert to a numpy array and returns it in the form of [num_images,size,size,channel]
        return np.asarray(images)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Testing Network')
    parser.add_argument('--in_dir',dest='in_dir',type=str,default='temp_images_test')
    parser.add_argument('--meta_file',dest= 'meta_file',type=str,default= r'D:\School Stuff\CBASE\Master_Project\Pre-Built_AI\Use_this_one\meta_file')
    parser.add_argument('--CP_dir',dest= 'chk_point_dir',type=str,default= r'D:\School Stuff\CBASE\Master_Project\Pre-Built_AI\Use_this_one\meta_file')
    parser.add_argument('--save_dir', dest = 'save_dir', type=str,default= r'D:\School Stuff\CBASE\Master_Project\Pre-Built_AI\Use_this_one\Output_Images')
    return parser.parse_args()
    ### adding
    print('save_dir = ', save_dir)

def main(args):
    #File names are saved into a cache file
    args=parse_arguments()
    dataset_test = cache(cache_path='my_dataset_cache_test.pkl',
                    fn=Dataset_test,
                    in_dir=args.in_dir)
    test_images = dataset_test.images

    graph = tf.Graph()
    with graph.as_default():
        with tf.compat.v1.Session() as sess:
            #import the model dir
            try:
                ### added ''
                file_=Path('args.meta_file')
                abs_path=file_.resolve()
            except FileNotFoundError:
                sys.exit('Meta File Not found')
            else:
                ### was imported_meta = tf.train.import_meta_graph(args.meta_file)
                imported_meta = tf.train.import_meta_graph(r'D:\School Stuff\CBASE\Master_Project\Pre-Built_AI\Use_this_one\meta_file\model.meta',
                                                           r'D:\School Stuff\CBASE\Master_Project\Pre-Built_AI\Use_this_one\meta_file\model_complete.meta')

            ### had args.chk_point_dir
            if os.path.isdir(r'D:\School Stuff\CBASE\Master_Project\Pre-Built_AI\Use_this_one\meta_file'):
                imported_meta.restore(sess, tf.train.latest_checkpoint(r'D:\School Stuff\CBASE\Master_Project\Pre-Built_AI\Use_this_one\meta_file'))
            else:
                sys.exit("Check Point Directory does not exist")

            x = graph.get_operation_by_name("x").outputs[0]
            predictions = graph.get_operation_by_name("predictions").outputs[0]
            print('x=',x,'pred=',predictions)

            #Take one image at a time, pass it through the network and save it
            for counter,image in enumerate(test_images):
                broken_image,h,w,h_no,w_no = break_image(image,128)

                output_image = np.zeros((h_no*128,w_no*128,3),dtype = np.uint8)

                feed_dict = {x: broken_image}
                batch_predictions = sess.run(predictions, feed_dict = feed_dict)

                matrix_pred = batch_predictions.reshape((h_no,w_no))
                #Concentrate after this for post processing
                for i in range(0,h_no):
                    for j in range(0,w_no):
                        a = matrix_pred[i,j]
                        output_image[128*i:128*(i+1),128*j:128*(j+1),:] = 1-a


                cropped_image = image[0:h_no*128,0:w_no*128,:]
                pred_image = np.multiply(output_image,cropped_image)

                print("Saved {} Image(s)".format(counter+1))
                cv2.imwrite(os.path.join(args.save_dir,'outfile_{}.jpg'.format(counter+1)), pred_image)
                ### adding
                print('save_dir = ', save_dir)
                print(pred_image), print(output_image)

if __name__ == '__main__':
    main(sys.argv)

test_img = Image.open(r'D:\School Stuff\CBASE\Master_Project\Crack7.jpg')
test_img2 = Image.open(r'D:\School Stuff\CBASE\Master_Project\temp_images\crack\Image_Crack1053.6r90.png')
#test_img.show()
break_image(test_img, 1000)
