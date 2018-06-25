"""
Mask R-CNN
Configurations and data loading code for the synthetic Shapes dataset.
This is a duplicate of the code in the noteobook train_shapes.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import math
import random
import numpy as np
import skimage
from mrcnn import model as modellib, utils

# Root directory of the project
ROOT_DIR = os.path.abspath("/home/ubuntu/Mask_RCNN/")
MODEL_DIR = os.path.join(ROOT_DIR,"logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils


class BuildingConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "buildings"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5



class BuildingDataset(utils.Dataset):

    #PATH = '/Users/tingold/code/Mask_RCNN/samples/buildings/training_data'
    PATH = os.path.join(ROOT_DIR,'samples/buildings/training_data')

    image_lookup = []

    def load_buildings(self,):

        self.add_class("buildings", 1, "building")
        print("Loading buildings")

        image_filenames = os.listdir(self.PATH + '/sat')
        cnt = 0
        for img_file in image_filenames:
            # id is the tile name without sat in front
            id = img_file.replace("sat","", 1)

            abs_img = self.PATH + "/sat/" + img_file
            self.image_lookup[cnt] = abs_img
            self.add_image("buildings", cnt, img_file)


    def load_mask(self, image_id):
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.

        print("Loading mask for image id "+self.image_lookup[image_id])


        mask_url = self.PATH+'/osm/osm_'+self.image_lookup[image_id]
        # Pack instance masks into an array
        mask = skimage.io.imread(mask_url, as_grey=True)
        return mask, 1



    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        print("Loading image for image id " + self.image_lookup[image_id])
        img_url = self.PATH + '/sat/sat' + self.image_lookup[image_id]
        # Pack instance masks into an array
        img = skimage.io.imread(img_url, as_grey=False)
        return img

if __name__ == '__main__':
    config = BuildingConfig()
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
    # Training
    dataset_train = BuildingDataset()
    dataset_train.load_buildings()
    dataset_train.prepare()
    # Validation
    dataset_val = BuildingDataset()
    dataset_val.load_buildings()
    dataset_val.prepare()

    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                layers='heads')