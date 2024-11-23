import transformers
import torch

from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8_bit=True)