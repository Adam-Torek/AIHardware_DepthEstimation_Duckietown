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

from transformers import HqqConfig, QuantoConfig

def get_curr_time_ms():
    return int(time.time()*1000)

def get_depth_from_model(input_image, image_processor, depth_model):
    inputs = image_processor(input_image, return_tensors="pt")
   
    with torch.no_grad():
        if inputs["pixel_values"].device != depth_model.device:
            inputs["pixel_values"] = inputs["pixel_values"].to(depth_model.device)
        start_time = get_curr_time_ms()
        outputs = depth_model(**inputs)
        end_time = get_curr_time_ms() - start_time
        predicted_depth = outputs.predicted_depth
    return predicted_depth, end_time

def collect_nyu2_results(nyu2_dataset, image_processor, depth_model):
    absrel_scores = []
    d1_scores = []
    average_inference_ms = []
    for i in range(nyu2_dataset.__len__()):
        input_image, depth_target = nyu2_dataset.__getitem__(i)
        
        input_image = input_image.to(depth_model.device)
        depth_target = depth_target.to(depth_model.device)
        
        predicted_depth, inference_ms = get_depth_from_model(input_image, image_processor, depth_model)
        average_inference_ms.append(inference_ms)

        valid_mask = [predicted_depth >= 0.0001]

        eval_results = eval_depth(predicted_depth[valid_mask], depth_target[valid_mask])
        absrel_scores.append(eval_results["abs_rel"])
        d1_scores.append(eval_results["d1"])

    return (sum(absrel_scores) / len(absrel_scores), 
            sum(d1_scores) / len(d1_scores), 
            sum(average_inference_ms) / len(average_inference_ms))

def collect_da2K_results(da2k_dataset, image_processor, depth_model):
    closer_match = []
    average_inference_ms = []
    for i in range(da2k_dataset.__len__()):
        da2k_image_data = da2k_dataset.__getitem__(i)
        input_image = da2k_image_data["image"]
        resizer = transforms.Resize(size=input_image.size)
        predicted_depth, inference_ms = get_depth_from_model(input_image, image_processor, depth_model)
        predicted_depth = resizer(predicted_depth).squeeze().cpu()
        average_inference_ms.append(inference_ms)
        closer_match.append(eval_accuracy(predicted_depth, da2k_image_data["points"], da2k_image_data["closer_point"]))


    return (sum([1 if x is True else 0 for x in closer_match]) / len(closer_match), 
            sum(average_inference_ms) / len(average_inference_ms))

def run_nyu_da2K_quantization_tests(model_weights, quant_configs, nyuv2_data, da_data):
    quant_trial_results = []
    for config_name, config in quant_configs.items():
        torch._C._cuda_clearCublasWorkspaces()
        image_processor = AutoImageProcessor.from_pretrained(model_weights, quantization_config=config, device_map="auto")
        depth_model = AutoModelForDepthEstimation.from_pretrained(model_weights, quantization_config=config, device_map="auto")
        model_size = torch.cuda.memory_allocated() / 1024**2

        abs_rel, d1, nyu2_time = collect_nyu2_results(nyuv2_data, image_processor, depth_model)
        accuracy, da2k_time = collect_da2K_results(da_data, image_processor, depth_model)
        quant_trial_results.append({ "config_name":config_name,
                                       "abs_rel":abs_rel, 
                                       "d1":d1, 
                                       "accuracy":accuracy, 
                                       "nyu2_time":nyu2_time, 
                                       "da2k_time":da2k_time, 
                                       "model_size":model_size
                                    })
    
    return quant_trial_results

def run_nyu2_da2k_on_all_models(output_directory, model_list, qauntization_configs, nyuv2_dataset, da2k_dataset):
    for model_weights in model_list:
        results_list_dict = run_nyu_da2K_quantization_tests(model_weights, 
                                                                    qauntization_configs, 
                                                                    nyuv2_dataset, 
                                                                    da2k_dataset)


        output_file_name = model_weights.split("/")[1]
        with open(os.path.join(output_directory, output_file_name + "_results.json"), "w") as json_output:
            json.dump(results_list_dict, json_output)


        with open(os.path.join(output_directory, output_file_name + "_results.csv"), "w") as csv_output:   
            writer = csv.DictWriter(csv_output, fieldnames=["config_name",
                                                            "abs_rel",
                                                            "d1",
                                                            "accuracy",
                                                            "nyu2_time",
                                                            "da2k_time",
                                                            "model_size"])
            writer.writeheader()
            writer.writerows(results_list_dict)

def main():
    input_parser = argparse.ArgumentParser(prog="evaluation_runner.py", description="")
    input_parser.add_argument("--nyu2_data_dir", help="location of the nyu2 depth dataset directory")
    input_parser.add_argument("--da2k_data_dir", help="location of the DA2K benchmark dataset directory")
    input_parser.add_argument("--output_dir", help="location of the directory to write the results to")

    try:
        input_args = input_parser.parse_args()
        nyu2_data_dir = input_args.nyu2_data_dir
        da2k_data_dir = input_args.da2k_data_dir
        output_dir = input_args.output_dir

        nyu2_dataset = NYUV2DataSet(nyu2_data_dir, "nyu2_test.csv")
        da2k_dataset = DA2KDataSet(da2k_data_dir)

        model_weights = ["depth-anything/Depth-Anything-V2-Small-hf",
                       "depth-anything/Depth-Anything-V2-Base-hf",
                       "depth-anything/Depth-Anything-V2-Large-hf"]
        
        quantization_configs = {"no_quant":None,
                                "hqq_8bit": HqqConfig(nbits=8, group_size=64),
                                "hqq_4bit": HqqConfig(nbits=4, group_size=64),
                                "hqq_2bit": HqqConfig(nbits=2, group_size=64),
                                "hqq_1bit": HqqConfig(nbits=1, group_size=64)}
        
        run_nyu2_da2k_on_all_models(output_dir, model_weights, quantization_configs, nyu2_dataset, da2k_dataset)

    except argparse.ArgumentError as e:
        print(e.message)
        input_parser.print_help()
        sys.exit(1)
        
    
if __name__ == "__main__":
    main()