import os
import glob
import re
import hashlib

from PIL import Image
import numpy as np
from tensorflow.python.util import compat


class GenInput:
    def __init__(self, dir="database", testingPercentage=10, validationPercentage=10, padding_colour="black"):
        self.dir = dir
        self.inputs = {"images": {"training":[], "testing":[],"validation":[]},
                       "labels": {"training":[], "testing":[],"validation":[]}}
        self.labels = []
        self.testingPercentage = testingPercentage
        self.validationPercentage = validationPercentage
        self.padding_colour = padding_colour
        self.createImageList()
        self.generate()
        self.setDimensions()
        print("Data loaded")

    def generate(self):
        # Possible errors:
        if len(self.labels) == 0:
            print("No valid folders of images found at " + self.dir + ".")
            return -1
        if len(self.labels) == 1:
            print("Only one valid folder of images found at " + self.dir + ". Multiple classes are needed for classification.")
            return -1

        i = 0
        dicLabels = {}
        for label in self.labels:
            dicLabels[label] = np.zeros(len(self.labels))
            dicLabels[label][i] = 1.
            i += 1

        biggest_image = (0,0)
        for label in self.labels:
            for category in self.imageList[label].keys():
                if category == "dir": continue
                categoryList = []
                for image in self.imageList[label][category]:
                    img = Image.open(image)
                    if img.size > biggest_image: biggest_image = img.size

        padding_size = (max(biggest_image) + 1, max(biggest_image) + 1) 

        for label in self.labels:
            for category in self.imageList[label].keys():
                if category == "dir": continue
                categoryList = []
                for image in self.imageList[label][category]:
                    img = Image.open(image)
                    # PLACE HERE PREPROCESSING FROM PIL
                    img = self.padding(img, padding_size)
                    img = self.changeContrast(img, 100)
                    imgArray = np.array(img)
                    # PLACE HERE PREPROCESSING FROM NUMPY
                    imgArray = imgArray.astype('float32')
                    imgArray /= 255
                    if len(imgArray.shape) == 2:
                        img_x, img_y = imgArray.shape
                        imgArray = imgArray.reshape(img_x, img_y, 1)
                    self.inputs["images"][category].append(imgArray)
                    self.inputs["labels"][category].append(dicLabels[label])
        
        for type in self.inputs.keys():
            for category in self.inputs[type].keys():
                self.inputs[type][category] = np.asarray(self.inputs[type][category])
                
    def changeContrast(self, img, level):
        factor = (259 * (level + 255)) / (255 * (259 - level))
        def contrast(c):
            return 128 + factor * (c - 128)
        return img.point(contrast)

    def padding(self, img, padding_size):
        new_image = Image.new("RGB", padding_size, self.padding_colour)
        a = (int((padding_size[0]-img.size[0])/2), int((padding_size[1]-img.size[1])/2))
        #print(a)
        new_image.paste(img, a)
        return new_image

    def createImageList(self):
        self.imageList = {}
        subDirs = [x[0] for x in os.walk(self.dir)]
        
        for subDir in subDirs:
            if subDir == self.dir:
                continue
            
            extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
            fileList = []
            dirName = os.path.basename(subDir)
            print("Looking for images in '" + dirName + "'")
            
            for extension in extensions:
                fileGlob = os.path.join(self.dir, dirName, '*.' + extension)
                fileList.extend(glob.glob(fileGlob))
            if not fileList:
                print("No files found")
                continue
            labelName = re.sub(r'[^a-z0-9]+', ' ', dirName.lower())
            self.labels.append(labelName)

            trainingImages = []
            testingImages = []
            validationImages = []

            for fileName in fileList:
                hashName = re.sub(r'_nohash_.*$', '', fileName)
                hashNameHashed = hashlib.sha1(compat.as_bytes(hashName)).hexdigest()
                percentageHash = (int(hashNameHashed, 16) % (65536)) * (100 / 65535.0)
                
                if percentageHash < self.validationPercentage:
                    validationImages.append(fileName)
                elif percentageHash < (self.testingPercentage + self.validationPercentage):
                    testingImages.append(fileName)
                else:
                    trainingImages.append(fileName)
            
            self.imageList[labelName] = {
                'dir': dirName,
                'training': trainingImages,
                'testing': testingImages,
                'validation': validationImages,
            }

    def setDimensions(self):
        self.dimensions = self.inputs["images"]["training"][0].shape
