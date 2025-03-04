import os
import sys

sys.path.append(os.path.abspath('/home/wyz/deeplearning/workspace/Privacy-USL-LLM'))
import logging.handlers
import torch
from typing import Union
from peft import get_peft_model, LoraConfig, TaskType

from dualguard.usl import *
from dualguard.utils.model import load_model_and_tokenizer

HeadModel=Union[QwenHead,LlamaHead,GPT2Head]
TailModel=Union[QwenTail,LlamaTail,GPT2Tail]
ServerModel=Union[QwenServer,LlamaServer,GPT2Server]

pt_model,tokenizer=load_model_and_tokenizer('gpt/gpt2-large')
split_config=SplitModelConfig(head_layer_num=3,server_layer_num=-1,tail_layer_num=3,with_server=True,logicl_load=True)
split_model=GPT2SplitModel(pt_model,split_config)

lora_config=LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=2,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["attn.c_proj", "attn.c_attn"]
)
print(split_model.head_model)
split_model.head_model=torch.load('/home/wyz/deeplearning/workspace/Privacy-USL-LLM/dualguard/static/warmup/a.pth',weights_only=False,map_location='cuda:0')
print(split_model.head_model)
# split_model.head_model=get_peft_model(split_model.head_model,peft_config=lora_config)
# torch.save(split_model.head_model,'/home/wyz/deeplearning/workspace/Privacy-USL-LLM/dualguard/static/warmup/a.pth')
# split_model.tail_model=get_peft_model(split_model.tail_model,peft_config=lora_config)
# print(split_model.head_model.)
# split_model.head_model.save_pretrained('/home/wyz/deeplearning/workspace/Privacy-USL-LLM/dualguard/static/warmup')
# print(split_model.head_model.)