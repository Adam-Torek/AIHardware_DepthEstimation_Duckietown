#!/usr/bin/env python3

import os
import rospy
import torch
import time
import numpy as np

from dpt import DepthAnythingV2

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage

import cv2
from cv_bridge import CvBridge

class CameraReaderNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        # static parameters
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        # bridge between OpenCV and ROS
        self._bridge = CvBridge()
        # create window for the input image
        self._window = "input_image"
        cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)

        # Create a window for the output depth estimation image
        self._output_window = "depth_image"
        cv2.namedWindow(self._output_window, cv2.WINDOW_AUTOSIZE)

        # set the desired input size of the image
        self.input_size = 518

        # Set the device depth-anything-v2 will run on
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        # Set the size of the depth estimation model
        self.encoder_type = 'vitb'

        # Set parameters for each of the depth estimation model sizes
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        # Load the depth estimation model into memory and its weights
        self.dpt2 = DepthAnythingV2(**self.model_configs[self.encoder_type])
        self.dpt2.load_state_dict(torch.load(f"models/depth_anything_v2_{self.encoder_type}.pth", map_location="cpu"))

        # Move the model into the specific device's memory and evaluate it 
        self.dpt2 = self.dpt2.to(self.device).eval()

        # construct subscriber
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)


    def callback(self, msg):
        time.sleep(0.01)
        # convert JPEG bytes to CV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg)

        # Run depth estimation on the given input image
        depth_image = self.dpt2.infer_image(image, input_size=self.input_size)

        # Normalize the depth image and convert it to unsigned 8 bit format
        depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min()) * 255.0
        depth_image = depth_image.astype(np.uint8)

        # Convert the depth image into an openCV2 image
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)

        # display frame
        cv2.imshow(self._window, image)
        cv2.imshow(self._output_window, depth_image)
        cv2.waitKey(1)

if __name__ == '__main__':
    # create the node
    node = CameraReaderNode(node_name='camera_reader_node')
    # keep spinning
    rospy.spin()