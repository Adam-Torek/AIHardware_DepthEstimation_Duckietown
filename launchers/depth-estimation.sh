#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber
rosrun depth_estimation camera_reader_node.py

# wait for app to end
dt-launchfile-join