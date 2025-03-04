import os,sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))[:-4]) # add the path of Privacy-USL-LLM to sys.path
# import usl.chatglm 
from transformers import Qwen2ForCausalLM,Qwen2Tokenizer,LlamaForCausalLM
from usl.qwen.qwen_split import QwenSplitModel
from usl.split_config import SplitModelConfig,Intermediate

import torch
model_path='Qwen/qwen2-1.5b'

tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
qwen = Qwen2ForCausalLM.from_pretrained(model_path)
print(qwen)

# split_qwen=QwenSplitModel(qwen,SplitModelConfig(2,24,2,True,True))
#split_qwen.head_model 获取头部模型
#split_qwen.tail_model 获取尾部模型

text='''better accuracy. But it is difficult to continue the trend to increase
model size due to limited GPU memory. One promising solution
is to support swapping between GPU and CPU memory. However,
existing work on swapping only handle certain models and do not
achieve satisfactory performance.
Deep learning computation is commonly expressed as a dataflow
graph which can be analyzed to improve swapping. We propose
SwapAdvisor, which performs joint optimization along 3 dimensions based on a given dataflow graph: operator scheduling, memory allocation, and swap decisions. SwapAdvisor explores the vast
search space using a custom-designed genetic algorithm. Evaluations using a variety of large models show that SwapAdvisor can
train models up to 12 times the GPU memory limit while achieving
53-99% of the throughput of a hypothetical baseline with infinite
GPU memory'''
tokenizer.pad_token = tokenizer.eos_token
inputs=tokenizer.encode(text, max_length=128, pad_to_max_length=True, return_tensors='pt')
print(f'inputs: {inputs.shape}')
output=qwen(input_ids=inputs,labels=inputs,with_server=True)
output.loss.backward(retain_graph=True)
# print(split_qwen.intermediate) # 获取中间结果

