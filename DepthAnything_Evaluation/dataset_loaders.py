from functools import partial

from PIL import Image
import torch
import os
import csv
import json
import numpy as np
from torchvision import transforms

"""Tensorflow-based data loader for the NYU2 depth image dataset. This utility
class lets us load both our test and training datasets using two or three lines
of code in two or three lines. This class includes image augmentation functionality
that will randomly flip and crop images and will always resize the image if a different
size is provided. """
class NYUV2DataSet(torch.utils.data.Dataset): 
    def __init__(self, dataset_path, csv_name, image_size = (518, 518)):
        # initialize the data loader parameters 
        self.dataset_path = dataset_path
        self.input_images = []
        self.depth_images = []
        self.image_size = image_size

        self.converter = transforms.Compose([
                    transforms.PILToTensor(),
                    transforms.Resize(self.image_size, antialias=True)
            ])

        # Open the CSV file that contains the paths to the input and 
        # depth estimation images 
        with open(os.path.join(dataset_path, csv_name)) as training_csv:
            csv_content = csv.reader(training_csv, delimiter=',')
            # Assemble the file paths for the input and depth images 
            for row in csv_content:
                input_image_path = os.path.join(*(row[0].split(os.path.sep)[1:]))
                depth_image_path = os.path.join(*(row[1].split(os.path.sep)[1:]))
                self.input_images.append(os.path.join(self.dataset_path,input_image_path))
                self.depth_images.append(os.path.join(self.dataset_path,depth_image_path))

        # Get the length of the dataset
        self.dataset_size = len(self.input_images)

    """Get the length of the NYU2 dataset."""
    def __len__(self):
        return self.dataset_size
    
    """Get an input image and depth image from the NYU2 data loader."""
    def __getitem__(self, idx):
        # Read the input image from the file path using OpenCV
        image =  Image.open(self.input_images[idx])
        # Load in the grayscale depth image from its file path using openCV
        depth_image = Image.open(self.depth_images[idx])

        pytorch_image = self.converter(image)
        pytorch_depth_image = self.converter(depth_image).float()

        return pytorch_image, pytorch_depth_image


class DA2KDataSet(torch.utils.data.Dataset):
    """Pytorch-based Data Loader for the DA-2K dataset 
    benchmark released by Yang et. al. This dataset contains
    2K real and generated images with a closer point and further
    point that are both annotated by humans. The closer and further
    away points are in challenging locations of the image that 
    require any model tested on said benchmark to precisely
    estimate the given depth of an image. They are also in varied
    locations covering many different scenes to test a depth 
    estimation model's ability to correctly determine depth in
    fine detail across a variety of situations, locations, and
    environments. The DA-2K images are made to be challenging 
    but represent realistic scenarios in which a depth estimation
    model may be used."""
    def __init__(self, dataset_path):
        """Initialize the dataset from the given path. Loads the images and
        point labels into memory. The images come by themselves from subfolders
        and the annotations come from a JSON file provided with the dataset."""
        self.dataset_path = dataset_path

        # Open the annotations JSON file and load its information into memory
        with open(os.path.join(dataset_path, "annotations.json"),"r") as annotations_file:
            image_annotations = json.load(annotations_file)
            # Get all of the image paths in the DA2K dataset
            self.image_paths = list(image_annotations.keys())

            # Set up class variables
            self.image_points = []
            self.closer_points = []
            self.images = []
            # Load in each image and labeled points into memory
            for i, image in enumerate(image_annotations.values()):
                image = image[0]
                # Load the locations of the two labelled points in
                self.image_points.append((image["point1"], image["point2"]))
                # Find which point is closer and label it
                if image["closer_point"] == "point1":
                    self.closer_points.append(0)
                elif image["closer_point"] == "point2":
                    self.closer_points.append(1)
                # Open the image as a PIL image file and load it into memory
                image_data = Image.open(os.path.join(self.dataset_path, self.image_paths[i]))
                self.images.append(image_data)
        # Set the length of the entire dataset
        self.len = len(self.image_paths)
        

    def __len__(self):
        """Return the size of the DA-2K dataset (around 2K images)"""
        return self.len

    def __getitem__(self, idx):
        """Get the image, the location of the two annotated points, 
        and which point is the closer point at the specified index."""
        image_data = {}
        image_data["image"] = self.images[idx]
        image_data["points"] = self.image_points[idx]
        image_data["closer_point"] = self.closer_points[idx]
        
        return image_data
