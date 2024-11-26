import argparse
import time
import json
import csv
import torch
import os
import sys

from loss import eval_depth, eval_accuracy
from dataset_loaders import NYUV2DataSet, DA2KDataSet

from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from transformers import HqqConfig

def get_curr_time_ms():
    """Get the current system time
    in milliseconds"""
    return int(time.time()*1000)

def get_depth_from_model(input_image, image_processor, depth_model):
    """Get a depth estimation from the Depth Anything model given
    an input image, a tokenizer for the image, and the Depth Anything
    Model itself"""

    # Get a tokenized image using the input processor and return it
    # as a PyTorch tensor
    inputs = image_processor(input_image, return_tensors="pt")

    # Turn off automatic gradient building in PyTorch for 
    # inferencing with the model
    with torch.no_grad():
        # Move the input image to the device the Depth Anything model
        # is on if it is not already there
        if inputs["pixel_values"].device != depth_model.device:
            inputs["pixel_values"] = inputs["pixel_values"].to(depth_model.device)
        # Time the model inference in milliseconds
        start_time = get_curr_time_ms()
        # Get a depth estimation from the tokenized image using 
        # the Depth Anything Model
        outputs = depth_model(**inputs)
        # Record the total inference time of the model
        end_time = get_curr_time_ms() - start_time
        # Get the depth prediction and return it along with
        # the total inference time
        predicted_depth = outputs.predicted_depth
    return predicted_depth, end_time

def collect_nyu2_results(nyu2_dataset, image_processor, depth_model):
    """Evaluate the NYUV2 testing dataset on the provided Depth Anything
    model and its associated image processor, and return the average
    abs-rel, d1 scores, and inference time for the Depth Anything model
    on the dataset by comparing the depth estimation from the model
    to the ground truth. """
    absrel_scores = []
    d1_scores = []
    average_inference_ms = []
    # Iterate over each input image and depth ground truth
    # from the NYUV2 testing datset. 
    for i in range(nyu2_dataset.__len__()):
        # Get the input image and depth ground truth at index 
        # i from the data loader
        input_image, depth_target = nyu2_dataset.__getitem__(i)
        
        # Send the input and depth image to the same PyTorhc device
        # that the model is running in
        input_image = input_image.to(depth_model.device)
        depth_target = depth_target.to(depth_model.device)
        
        # Get the predicted depth estimation and 
        # the inference time of the Depth Anything model
        predicted_depth, inference_ms = get_depth_from_model(input_image, image_processor, depth_model)
        # Record the inference time for this image
        average_inference_ms.append(inference_ms)
        # Prevent non-zero depth estimation values from being used
        # in the eval depth metrics
        valid_mask = [predicted_depth >= 0.0001]

        # Get the abs_rel and d1 metric scores for the depth estimation
        # of the single image
        eval_results = eval_depth(predicted_depth[valid_mask], depth_target[valid_mask])
        absrel_scores.append(eval_results["abs_rel"])
        d1_scores.append(eval_results["d1"])

    # Return the average absrel, d1, and inference time scores
    # for all NYUV2 test images for the given model
    return (sum(absrel_scores) / len(absrel_scores), 
            sum(d1_scores) / len(d1_scores), 
            sum(average_inference_ms) / len(average_inference_ms))

def collect_da2K_results(da2k_dataset, image_processor, depth_model):
    """Get the average closest-pixel accuracy for the given Depth Anything model 
    and image processor for the DA2K dataset, as well as the average inference 
    time in milliseconds it took to complete. """
    closer_match = []
    average_inference_ms = []
    # Iterate over each image in the DA2K dataset
    for i in range(da2k_dataset.__len__()):
        # Get the image and the two labelled points for accuracy classification
        da2k_image_data = da2k_dataset.__getitem__(i)
        # Get the input image for depth estimation
        input_image = da2k_image_data["image"]
        # Get the predicted depth estimation map and inference time
        # for the depth anything model
        predicted_depth, inference_ms = get_depth_from_model(input_image, image_processor, depth_model)
        # Resize the depth estimation map output to be the correct size 
        resizer = transforms.Resize(size=input_image.size)
        predicted_depth = resizer(predicted_depth).squeeze().cpu()
        # record the inference time and whether or not the depth estimation map
        # from the Depth Anything model was the closer of the two annotated pixel
        average_inference_ms.append(inference_ms)
        closer_match.append(eval_accuracy(predicted_depth, da2k_image_data["points"], da2k_image_data["closer_point"]))

    # Calculate the average accuracy score across all DA-2K images and 
    # the average inference time for the DA2K dataset
    return (sum([1 if x is True else 0 for x in closer_match]) / len(closer_match), 
            sum(average_inference_ms) / len(average_inference_ms))

def run_nyu_da2K_quantization_tests(model_weights, quant_configs, nyuv2_data, da_data):
    """Run both the NYUV2 and DA-2K test benchmarks on the provided Depth Anything model
    across various quantization configurations. This function will record the abs_rel scores,
    d1 scores, and inference times for the NYUV2 and DA-2K test benchmarks on each quantization
    configuration, as well as the total allocated size of the Depth Anything model in megabytes."""
    quant_trial_results = []
    # Iterate over each quantization configuration, initialize the given Depth Anything
    # model with that configuration, and run the quantized Depth Anything V2 model on
    # the NYU-V2 and DA-2K benchmarks to collect results
    for config_name, config in quant_configs.items():
        # Clear the memory allocation of the last Depth Anything V2 trial
        torch._C._cuda_clearCublasWorkspaces()
        # Initialize the image preprocessor for the Depth Anything V2 model using 
        # the provided quantization configuration and automatically select 
        # the best PyTorch device to run it on (The GPU if available, the CPU if no GPU)
        image_processor = AutoImageProcessor.from_pretrained(model_weights, quantization_config=config, device_map="auto")
        # Initialize a pretrained Depth Anything V2 model based on the given weights
        # with the given quantization configuration and  automatically select the 
        # best PyTorch device to use (usually the GPU or the CPU if no GPU is detected)
        depth_model = AutoModelForDepthEstimation.from_pretrained(model_weights, quantization_config=config, device_map="auto")
        # get the total size of the memory allocation required for the Depth Anything V2 model
        # in megabytes
        model_size = torch.cuda.memory_allocated() / 1024**2
        # Get the average abs_rel and d1 scores for the quantized Depth Anything V2 model 
        # as well as the average inference time for the NYUV2 test benchmark
        abs_rel, d1, nyu2_time = collect_nyu2_results(nyuv2_data, image_processor, depth_model)
        # Get the average accuracy and inference time for the quantized Depth Anything V2 Model
        # on the DA-2K benchmark dataset
        accuracy, da2k_time = collect_da2K_results(da_data, image_processor, depth_model)
        # Collect all of the metrics into a convenient list of dictionary values for this
        # Depth Anything V2 quantization trial for later use 
        quant_trial_results.append({ "config_name":config_name,
                                       "abs_rel":abs_rel, 
                                       "d1":d1, 
                                       "accuracy":accuracy, 
                                       "nyu2_time":nyu2_time, 
                                       "da2k_time":da2k_time, 
                                       "model_size":model_size
                                    })
    # Return the results once all quantization trials are complete
    return quant_trial_results

def run_nyu2_da2k_on_all_models(output_directory, model_list, qauntization_configs, nyuv2_dataset, da2k_dataset):
    """Run all of the provided quantization configurations with the given list of Depth Anything V2 model weights
    and record the trial results to disk. This function takes N Depth Anything V2 pretrained model weights
    and M quantization configurations to run N x M trials total. This function uses the provided NYUV2 and
    DA2K datasets for each trial and records the results to disk once they are complete. Each trial's results
    are saved in both JSON and CSV output to the specified directory. One JSON and CSV file is created for
    every provided set of pretrained Depth Anything V2 model sizes, and each JSON and CSV file contains the results
    from all of the provided quantization configurations for that model size. Each trial contains the results
    of the model's performance on the NYUV2 test benchmark and DA-2K benchmark."""
    # Iterate over every set of pretrained Depth Anything V2 size in the provided
    # list of Depth Anything V2 model size
    for model_weights in model_list:
        # Run all of provided quantization configuration trials on this 
        # particular Depth Anything V2 model size with the NYUV2 and DA-2K 
        # benchmarks
        results_list_dict = run_nyu_da2K_quantization_tests(model_weights, 
                                                                    qauntization_configs, 
                                                                    nyuv2_dataset, 
                                                                    da2k_dataset)

        # Get the name of the results file from the Depth Anything V2's model size name
        output_file_name = model_weights.split("/")[1]
        # Save the results of all M quantization trials with this model in JSON format to
        # the provided output directory
        with open(os.path.join(output_directory, output_file_name + "_results.json"), "w") as json_output:
            # Use the Python JSON library to write the results to the file pointer
            json.dump(results_list_dict, json_output)

        # Save the results of all quantization trials with this model to CSV format to the provided
        # output directory
        with open(os.path.join(output_directory, output_file_name + "_results.csv"), "w") as csv_output:   
            writer = csv.DictWriter(csv_output, fieldnames=["config_name",
                                                            "abs_rel",
                                                            "d1",
                                                            "accuracy",
                                                            "nyu2_time",
                                                            "da2k_time",
                                                            "model_size"])
            # Write the header of each column for each metric to disk
            writer.writeheader()
            # Save all of the results to the CSV file on the disk
            writer.writerows(results_list_dict)

def main():
    """Main function that parses command line arguments 
    before running M quantization trials on N different Depth Anything V2 model sizes
    and configurations using the NYUV2 and DA-2K datasets. 
    See run_nyu2_da2k_on_all_models for more details."""
    # set up the argument parser for reading in the directories of the NYUV2 and DA-2K 
    # benchmarks as well as the output directory to save the results to
    input_parser = argparse.ArgumentParser(prog="evaluation_runner.py", description="")
    input_parser.add_argument("--nyu2_data_dir", help="location of the nyu2 depth dataset directory")
    input_parser.add_argument("--da2k_data_dir", help="location of the DA2K benchmark dataset directory")
    input_parser.add_argument("--output_dir", help="location of the directory to write the results to")

    # Try parsing the arguments and running the trials
    try:
        input_args = input_parser.parse_args()
        # Get the NYUV2, DA-2K and output directories from the 
        # argument parser if parsing was successful
        nyu2_data_dir = input_args.nyu2_data_dir
        da2k_data_dir = input_args.da2k_data_dir
        output_dir = input_args.output_dir

        # Load the NYUV2 and DA-2K datasets into memory
        nyu2_dataset = NYUV2DataSet(nyu2_data_dir, "nyu2_test.csv")
        da2k_dataset = DA2KDataSet(da2k_data_dir)

        # Set up the different Depth Anything V2 model weights and 
        # configurations to run trials on
        model_weights = ["depth-anything/Depth-Anything-V2-Small-hf",
                       "depth-anything/Depth-Anything-V2-Base-hf",
                       "depth-anything/Depth-Anything-V2-Large-hf"]
        
        # Set up all of the possible quantization configurations to be used
        # on each model. Note that they all currently use the Half-Quadratic
        # Quantization (HQQ) library for quantization. This linear-like
        #  post-training quantization technique loads both the model 
        # weights and activations into N-bit integer ranges. 
        quantization_configs = {"no_quant":None,
                                "hqq_8bit": HqqConfig(nbits=8, group_size=64),
                                "hqq_4bit": HqqConfig(nbits=4, group_size=64),
                                "hqq_2bit": HqqConfig(nbits=2, group_size=64),
                                "hqq_1bit": HqqConfig(nbits=1, group_size=64)}
        
        # Run all possible N x M trials using N Depth Anything V2 model sizes
        # on M quantization configurations and save the results to disk.
        run_nyu2_da2k_on_all_models(output_dir, model_weights, quantization_configs, nyu2_dataset, da2k_dataset)

    # Throw an exception if the arguments were not able to 
    # be parsed. Print 
    except argparse.ArgumentError as e:
        # Print the error message received
        print(e.message)
        # Print the help message for the parser
        input_parser.print_help()
        # exit the program in an error state
        sys.exit(1)
        
# This piece of code is meant to stop other Python
# programs and processes from continously spawning 
# new running instances from this program and crashing
# the computer it's running on.  
if __name__ == "__main__":
    main()