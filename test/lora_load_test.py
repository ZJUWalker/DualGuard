import os
import sys

sys.path.append(os.path.abspath('/home/wyz/deeplearning/workspace/Privacy-USL-LLM'))
import logging.handlers

from typing import Union
from peft import get_peft_model, LoraConfig, TaskType

from dualguard.usl import *
from dualguard.utils.model import load_model_and_tokenizer

from peft import AutoPeftModel,AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM
model= AutoPeftModelForCausalLM.from_pretrained('/home/wyz/deeplearning/workspace/Privacy-USL-LLM/dualguard/static/warmup')
print(model)