'''
Image treatment

    author: Bruno Smarsaro Bazelato
'''

import os
import glob
import re
import hashlib

import numpy as np
import tensorflow as tf

from tensorflow.python.util import compat
from tensorflow.python.platform import gfile

image_path = "teste.jpg"
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

class process:
    '''
    Attributes:
        path: path to the database
        labels: list of labels

    Methods:
        gen_labels: generate one file with the labels to the data and another file with the files' name and their respective label

    '''
    def __init__(self, path = "database"):
        self.path = path
        self.labels = []

    def set_path(self, path):
        self.path = path

    def gen_labels(self):
        if not gfile.Exists(self.path):
            print("Image directory '" + self.path + "' not found.")
            return None
        fp_files = open("files.txt", "a+")
        for r,d,folder in os.walk(self.path):
            self.labels.append(os.path.basename(r))
            for file in folder:
                fp_files.write("%s\t%s\n"%(file, os.path.basename(r)))

        fp_files.close()
        with open("trained_labels.txt", "a+") as fp_labels:
            for i in range(1, len(self.labels)):
                fp_labels.write(self.labels[i])
                fp_labels.write("\n")

    def create_image_lists(self, testing_percentage, validation_percentage):
        """Builds a list of training images from the file system.

          Analyzes the sub folders in the image directory, splits them into stable
          training, testing, and validation sets, and returns a data structure
          describing the lists of images for each label and their paths.

          Args:
            image_dir: String path to a folder containing subfolders of images.
            testing_percentage: Integer percentage of the images to reserve for tests.
            validation_percentage: Integer percentage of images reserved for validation.

          Returns:
            A dictionary containing an entry for each label subfolder, with images split
            into training, testing, and validation sets within each label.
        """
        result = {}
        sub_dirs = [x[0] for x in os.walk(self.path)]
        is_root_dir = True
        for sub_dir in sub_dirs:
            if is_root_dir:
                is_root_dir = False
                continue
            extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
            file_list = []
            dir_name = os.path.basename(sub_dir)
            if dir_name == self.path:
                continue
            print("Looking for images in '" + dir_name + "'")
            for extension in extensions:
                file_glob = os.path.join(self.path, dir_name, '*.' + extension)
                file_list.extend(glob.glob(file_glob))
            if not file_list:
                print('No files found')
                continue
            if len(file_list) < 20:
                print('WARNING: Folder has less than 20 images, which may cause issues.')
            label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
            training_images = []
            testing_images = []
            validation_images = []
            for file_name in file_list:
                base_name = os.path.basename(file_name)
                # We want to ignore anything after '_nohash_' in the file name when
                # deciding which set to put an image in, the data set creator has a way of
                # grouping photos that are close variations of each other. For example
                # this is used in the plant disease data set to group multiple pictures of
                # the same leaf.
                hash_name = re.sub(r'_nohash_.*$', '', file_name)
                # This looks a bit magical, but we need to decide whether this file should
                # go into the training, testing, or validation sets, and we want to keep
                # existing files in the same set even if more files are subsequently
                # added.
                # To do that, we need a stable way of deciding based on just the file name
                # itself, so we do a hash of that and then use that to generate a
                # probability value that we use to assign it.
                hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
                percentage_hash = (int(hash_name_hashed, 16) % (65536)) * (100 / 65535.0)
                if percentage_hash < validation_percentage:
                    validation_images.append(base_name)
                elif percentage_hash < (testing_percentage + validation_percentage):
                    testing_images.append(base_name)
                else:
                    training_images.append(base_name)
            result[label_name] = {
                'dir': dir_name,
                'training': training_images,
                'testing': testing_images,
                'validation': validation_images,
            }

        return result