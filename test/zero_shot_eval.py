import os
import sys
sys.path.append(os.path.abspath('/home/wyz/deeplearning/workspace/Privacy-USL-LLM'))


from dualguard.defense.dp_config import DPConfig
from dualguard.utils.configs import DatasetArgs, EnvArgs, USLTrainArgs
from dualguard.utils.exp import load_datasets
from dualguard.utils.model import load_model_and_tokenizer, set_random_seed
from dualguard.usl import *
from dualguard.experiment.method_config import *
from dualguard.experiment.train.usl_formal_train import evaluate_usl

from typing import Union

HeadModel=Union[QwenHead,LlamaHead,GPT2Head]
TailModel=Union[QwenTail,LlamaTail,GPT2Tail]
ServerModel=Union[QwenServer,LlamaServer,GPT2Server]


if __name__ == '__main__':
    set_random_seed(42)
    device='cuda:1'
    usl_args=USLTrainArgs()
    # load dataset
    pt_model,tokenizer=load_model_and_tokenizer('gpt/gpt2-large')
    dataset_args=DatasetArgs(dataset_name='e2e')
    data_loaders=load_datasets(dataset_args,tokenizer)
    train_data_loader=data_loaders['train']
    valid_data_loader=data_loaders['validation'] if 'validation' in data_loaders.keys() else data_loaders['test']
    split_model=GPT2SplitModel(pt_model,SplitModelConfig(3,-1,3,True,True),dp_config=DPConfig(add_noise=False))
    split_model.disable_dp()
    split_model.to(device)
    res=evaluate_usl(
        usl_args=usl_args,
        env_args=EnvArgs(device=device),
        head_model=split_model.head_model,
        server_model=split_model.server_model,
        tail_model=split_model.tail_model,
        valid_loader=valid_data_loader,
        tokenizer=tokenizer,
        output_similarity=True
    )
    print(res)
    
    
    
    