# CS 497/597 AI Hardware Final Project: Depth Estimation

## Team 2: The Convoluted
### Team Members:
- Adam Torek: adamtorek@u.boisestate.edu
- Niko Pedraza: nikopedraza@u.boisestate.edu
- Carson Keller: carsonkeller@u.boisestate.edu

## Project Purpose:

This project explores using the Depth-Anything-V2 monocular depth estimation model on
specialized hardware and studies the effect quantization plays on the model's performance. 
We used the Duckietown Robot as our hardware testing platform and Half-Quadratic Quantization
as our quantization scheme. For benchmarking Depth-Anything-V2 on our quantization settings, 
we used the DA-2K benchmark and the NYUV2 Depth Estimation Test datasets. We tested both 
the model's performance on these benchmarks as well as the average inference time and model
size to see the effect quantization has on both inference time and memory consumption. 

## Project Subsections:

This project contains two subsections: Our experimental evaluations and the 
duckiebot module. The experimental evaluations module can be found in the DepthAnythng_Evaluation
folder and the Duckiebot module that runs Depth Anything V2 can be found in the packages 
folder of this repository. Note: For our experimental evaluation, you will have to download
the NYUV2 and DA-2K datasets yourself from their respective websites. 

### DepthAnything V2 Evaluation 

This subsection of the project can be found in the DepthAnything_Evaluation subfolder of
this project. To run the quantization experiments, run the `quantization_runner.py` file
and provide it with the following command line arguments:

- `NYUV2 Dataset`: The directory where the NYU-2K dataset was stored

- `DA-2K Dataset`: The directory you stored the DA-2K benchmark

- `Output Directory`: The directory to write the experimental results to

Note: You will need to download both the NYUV2 and 
DA-2K datasets yourself. You can find them here:

- [NYUV2 Depth Dataset](https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2)

- [DA-2K Dataset](https://huggingface.co/datasets/depth-anything/DA-2K)

This experimental evaluation will run different quantization settings on the Depth Anything V2
Small, Medium, and Large models using the NYU-V2 and DA-2K datasets. Our evaluations are currently
set up to run 8-bit, 4-bit, 2-bit, and 1-bit quantization using the half-quadratic quantization
(HQQ) technique (More information can be found here: [Link](https://mobiusml.github.io/hqq_blog/)) as well as the unquantized version of each model as a baseline. Our experiments
save the results into both a JSON and CSV file for analysis. We use
the following metrics to evaluate our Depth-Anything-V2 model and all of the quantization
trials:

- Absolute Relative \(Abs-Rel\) Difference: The mean of the absolute difference between a depth
estimation image and a ground truth depth image per pixel. This is used to benchmark model
performance on the NYU-V2 dataset to measure both how well the Depth-Anything-V2 model performs
as well as if quantization has any effect on the model's performance. 

- Delta-1 (D1) Ratio: This is the ratio of how many pixels in a depth estimation deviate from
a given depth ground truth image by more than 25% relative to all of the pixels in the image.
Used to measure how much the Depth-Anything-V2 model's estimations differ from the ground truth
in the NYU-V2 test benchmark, as well as the effect quantization has on that difference.

- Point Classification Accuracy: Given a set of images with two human-annotated points where one
is closer than the other, how many did the model guess was closer given the depth estimation image?
This was used to evaluate how well the Depth-Anything-V2 model did on the DA-2K benchmark. This
benchmark includes images in a variety of scenes with points chosen by the human annotators that
require fine-grained depth estimations to guess correctly. This metric was used to test if
quantization of the Depth-Anything-V2 model had any effect on its fine-grained estimation performance.

- Inference Time in milliseconds: This is the total time it took the Depth-Anything-V2 model to 
create a depth estimation given an input image of a scene. We measured this metric to see if 
quantization had any effect on the time it took for the Depth-Anything-V2 model to
create its estimations. This is important for real-time applications that are time-sensitive.

- Model Size in Megabytes: This measures the total memory footprint of the Depth-Anything-V2
model. We recorded this metric to see how much we could compress our Depth-Anything-V2 model
using the 8-bit, 4-bit, 2-bit, and 1-bit quantization sizes we tested.

You can find both the results and our interpretation of those results in either the 
analysis section of the README or the `results_visualization.ipynb` notebook. We ran our model
on a desktop computer with a NVIDIA RTX 3090 GPU with 24GB of VRAM, a 32 core AMD processor, 
and 64 GB of RAM. We used an Anaconda environment with Python version 3.11, PyTorch version 2.5,
CUDA version 11.8, and the Transformers library of version 

### DuckieBot Package

The purpose of this subsection of our project was to evaluate a possible deployment platform
for the Depth-Estimation-V2 model. We chose the DuckieTown Robot (DuckieBot) as our deployment
platform for this purpose. The DuckieBot is both a robotic platform and ecosystem that gives
researchers a standardized platform to conduct robotics research and a convenient set of
tools for higher education to teach robotics and autonomous systems to students online and
in person. We used the DuckieBot because of its ease of use, low cost, and community support 
options in case something went wrong. 

For our hardware, we used the DuckieBot DB21M platform, which is a ground-based robot with a 
single camera and powered by an NVIDIA Jetson Nano 2GB development board. We used the Daffy 
version of the DuckieBot software platform and created our package on top of the DuckieBot's 
prebuilt Robot Operating System (ROS) toolset. We used the unquantized Depth-Anything-V2 
base model as our monocular depth estimation model in a Python ROS package and deployed our model
to the robot using a Docker container. 

To run our model, connect a DuckieBot DB21M to a host computer with Docker, Python, and Pip 
installed. You will need the DuckieTown shell to build and deploy our package to the robot.
Follow [these instructions](https://docs.duckietown.com/daffy/opmanual-duckiebot/setup/setup_laptop/index.html) 
to set up your DuckieBot and deployment environment.

Once your DuckieBot and development platform are set up,
you can run the following command to build your docker
container once you are in the project directory:

`$ dts devel build -f`

And then run the following command to run our package
on your DuckieBot:

`$ dts devel run -R <robot name> -L depth-estimation -X`

Once it is running, you should see two windows with 
a live camera feed on one and a real-time depth estimation
from the Depth-Anything-V2 base model on the other. 

## Results

TODO: Put your results analysis and tables here

## Citations

- Depth Anything V2 Paper: [Link](https://arxiv.org/abs/2406.09414) 

- Duckietown Robot Platform: [Link](https://duckietown.com/)

- NYUV2 Depth Dataset: [Link](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)

- HuggingFace (Model Repository): [Link](https://huggingface.co/)

- Half Quadratic Quantization (HQQ) Implementation: [Link](https://mobiusml.github.io/hqq_blog/)