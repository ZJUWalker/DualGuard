
import os
from dualguard.usl.llama.llama_split import *
from dualguard.usl.qwen.qwen_split import *
from dualguard.usl.gpt.gpt2_split import *
from dualguard.usl.split_config import SplitModelConfig
from dualguard.utils import env
from dualguard.utils.configs import LogArgs, USLTrainArgs, DatasetArgs,EnvArgs,USLTrainArgs,TAGAttackArgs,LampAttackArgs,BISRAttackArgs
from dualguard.utils.exp import load_datasets
from dualguard.attack.attack_eval import *
from dualguard.utils.logger import create_logger
from dualguard.utils.model import load_model_and_tokenizer, set_random_seed
from dualguard.experiment.method_config import *
import torch
from torch.utils.data import DataLoader

from typing import Union

from transformers import GPT2LMHeadModel
from peft import LoraConfig, TaskType

HeadModel=Union[QwenHead,LlamaHead,GPT2Head]
TailModel=Union[QwenTail,LlamaTail,GPT2Tail]
ServerModel=Union[QwenServer,LlamaServer,GPT2Server]

def load_needed_models(method_conf:MethodConfig,ds_args:DatasetArgs,env_args:EnvArgs,dp_config:DPConfig=None):
    #加载预训练模型的头部和尾部
    pt_model,tokenizer=load_model_and_tokenizer(method_conf.usl_args.model_name) #warmup训练不需要server
    split_model:Union[LlamaSplitModel,QwenSplitModel,GPT2SplitModel]=None
    simple_name=method_conf.usl_args.model_name.split('/')[-1]#用于日志和保存模型
    #拆分模型
    if isinstance(pt_model, GPT2LMHeadModel):
        split_model=GPT2SplitModel(pt_model,SplitModelConfig(3,-1,3,True,True),dp_config=dp_config)
    elif isinstance(pt_model, LlamaForCausalLM):
        split_model=LlamaSplitModel(pt_model,SplitModelConfig(2,-1,2,True,True),dp_config=dp_config)
    elif isinstance(pt_model, Qwen2ForCausalLM):
        split_model=QwenSplitModel(pt_model,SplitModelConfig(3,-1,3,True,True),dp_config=dp_config)
    else:
        raise ValueError("Unsupported model type")
    split_model.disable_dp()
    #加载模型
    method_prefix=method_conf.prefix
    model_dir=os.path.join(method_conf.usl_args.save_dir,simple_name,ds_args.dataset_name)
    if method_conf.dp_config.add_noise:
        # method_prefix+=f"_e_{dp_config.epsilon}"
        model_path=os.path.join(model_dir,f"{method_prefix}_e_{dp_config.epsilon}")
    else:
        model_path=os.path.join(model_dir,f"{method_prefix}")
    #如果是dualguard，则加载warmup后的head模型
    if method_prefix =='dualguard':
        wp_head_model_ph=os.path.join(method_conf.usl_args.wp_model_dir,simple_name,ds_args.dataset_name,"head.pth")
        head_model=torch.load(wp_head_model_ph,map_location=env_args.device,weights_only=False)
        #加载头部模型
        #特殊处理
    else:
        head_model=torch.load(f'{model_path}_head.pth',map_location=env_args.device,weights_only=False)
    server_model=torch.load(f"{model_path}_server.pth",map_location=env_args.device,weights_only=False)
    tail_model=torch.load(f"{model_path}_tail.pth",map_location=env_args.device,weights_only=False)     
    split_model.to(env_args.device)
    # pt_model.to(env_args.device)
    return head_model,server_model,tail_model,split_model.head_model,split_model.tail_model,tokenizer     

def dataset_len(dataloader:DataLoader):
    token_len=0
    for i,batch in enumerate(dataloader):
        token_len+=batch['input_ids'].shape[1]
    return {
        'length':len(dataloader),
        'avg_token_len':token_len/len(dataloader)
    }

total_args=[naive_config,dp_forward_config,dxp_config,dp_sgd_config,dualguard_config] #可拔插的配置
eval_methods=['sip_attack','tag_attack','lamp_attack','bisr_attack','connet_to_pt_tail'] #可拔插的评估方法

if __name__ == '__main__':

    #配置参数
    ds_args = DatasetArgs(dataset_name=GSM8K,batch_size=4)
    env_args = EnvArgs(device='cuda:0')
    set_random_seed(env_args.random_seed)
    target_model_name=GPT2_LARGE
    
    for args in total_args:
        usl_args=args.usl_args
        usl_args.lora_config=gpt_lora_config
        dp_config=args.dp_config
        method_prefix=args.prefix
        simple_name=usl_args.model_name.split('/')[-1]#用于日志和保存模型
        logger=create_logger(log_args=LogArgs(log_dir=env.eval_log_dir,
                                              log_file_name=f'usl_defense_eval_{simple_name}_on_{ds_args.dataset_name}.log'))
        if dp_config.add_noise:
            log_str=f"\n{'='*20} {args.info} on model {simple_name} dataset {ds_args.dataset_name} with dp epsilon: {dp_config.epsilon} {'='*20}\n"
        else:
            log_str=f"\n{'='*20} {args.info} on model {simple_name} dataset {ds_args.dataset_name} without dp {'='*20}\n"
        logger.info(log_str)
        print(log_str)
        # model_dir=f"{usl_args.save_dir}{simple_name}/{ds_args.dataset_name}/{method_prefix}"
        logger.info(f'Loading {simple_name} model and tokenizer...')
        #加载必要的模型
        try:
            head_model,server_model,tail_model,forzen_head_model,forzen_tail_model,tokenizer\
                =load_needed_models(usl_args,ds_args,env_args,dp_config)
            #加载数据集
            logger.info(f'Loading dataset {ds_args.dataset_name}...')
            data_loaders=load_datasets(ds_args,tokenizer=tokenizer)
            # train_data_loader=data_loaders['train']
            # print(f"train dataset length: {dataset_len(train_data_loader)}")
            valid_data_loader=data_loaders['validation'] if 'validation' in data_loaders.keys() else data_loaders['test']
            #开始评估
            if 'sip_attack' in eval_methods:
                #tag_attack
                #加载sip模型
                log_str=f'Evaluating sip attack... on model {simple_name} with dataset {ds_args.dataset_name}'
                logger.info(log_str)
                print(log_str)
                sip_abs_path = f'/home/wyz/deeplearning/workspace/Privacy-USL-LLM/experiment/sip/{simple_name}_sip_model.pth'
                sip_model=torch.load(sip_abs_path,map_location=env_args.device,weights_only=False)
                rouge_l_f,meteor,loss = sip_attack_evaluate(
                    env_args=env_args,
                    attack_model=sip_model,
                    lm_net_Head=head_model,
                    tokenizer=tokenizer,
                    valid_loader=valid_data_loader)
                log_str=f'sip attack on model {simple_name} on {ds_args.dataset_name} dataset, Rouge_Lf1: {rouge_l_f:.4f} | Meteor: {meteor:.4f}\n'
                logger.info(log_str)
                print(log_str)
                del sip_model
                torch.cuda.empty_cache()
            if 'tag_attack' in eval_methods:
                log_str=f'Evaluating tag attack... on model {simple_name} with dataset {ds_args.dataset_name}'
                logger.info(log_str)
                print(log_str)
                rouge_l_f,meteor=tag_attack_evaluate(
                    env_args=env_args,
                    tag_args=TAGAttackArgs(sample_num=1,attack_iters=500,is_warmup_usl=True) if method_prefix == 'warmup_usl' else TAGAttackArgs(sample_num=1,attack_iters=500),
                    head_model=head_model,
                    tail_model=tail_model,
                    forzen_tail_model=forzen_tail_model,
                    server_model=server_model,
                    tokenizer=tokenizer,
                    data_loader=valid_data_loader,
                )
                log_str=f"tag attack on model {simple_name} on attack method {method_prefix}  with dataset {ds_args.dataset_name} -> rouge-l f1: {rouge_l_f:.4f} meteor: {meteor:.4f}\n"
                logger.info(log_str)
                print(log_str)
                torch.cuda.empty_cache()
            if 'lamp_attack' in eval_methods:
                #lamp_attack
                log_str=f'Evaluating lamp attack... on model {simple_name} with dataset {ds_args.dataset_name}'
                logger.info(log_str)
                print(log_str)
                rouge_l_f,meteor=lamp_attack_evaluate(
                    env_args=env_args,
                    lamp_args=LampAttackArgs(sample_num=1,attack_iters=1500,lamp_interval=200,is_warmup_usl=True) if method_prefix == 'warmup_usl'\
                        else LampAttackArgs(sample_num=1,attack_iters=1500,lamp_interval=200),
                    pt_llm=None,
                    head_model=head_model,
                    tail_model=tail_model,
                    forzen_tail_model=forzen_tail_model,
                    server_model=server_model,
                    tokenizer=tokenizer,
                    data_loader=valid_data_loader,
                )
                log_str=f"lamp attack on model {simple_name} on attack method {method_prefix}  with dataset {ds_args.dataset_name} -> rouge-l f1: {rouge_l_f:.4f} meteor: {meteor:.4f}\n"
                logger.info(log_str)
                print(log_str)
                torch.cuda.empty_cache()
            # bisr_attack
            if 'bisr_attack' in eval_methods:    
                log_str=f'Evaluating bisr attack... on model {simple_name} with dataset {ds_args.dataset_name}'
                logger.info(log_str)
                print(log_str)
                # if sip_model is None:
                sip_abs_path = f'/home/wyz/deeplearning/workspace/Privacy-USL-LLM/experiment/sip/{simple_name}_sip_model.pth'
                sip_model=torch.load(sip_abs_path,map_location=env_args.device,weights_only=False)
                rouge_l_f,meteor=bisr_attack_evaluate(
                    env_args=env_args,
                    bisr_args=BISRAttackArgs(sample_num=1,attack_iters=1200,is_warmup_usl=True) if method_prefix == 'warmup_usl'\
                        else BISRAttackArgs(sample_num=1,attack_iters=1200),
                    sip_model=sip_model,
                    head_model=head_model,
                    tail_model=tail_model,
                    forzen_head_model=forzen_head_model,
                    forzen_tail_model=forzen_tail_model,
                    server_model=server_model,
                    tokenizer=tokenizer,
                    data_loader=valid_data_loader,
                )
                log_str=f"bisr attack on model {simple_name} on attack method {method_prefix} with dataset {ds_args.dataset_name} -> rouge-l f1: {rouge_l_f:.4f} meteor: {meteor:.4f}\n"
                logger.info(log_str)
                print(log_str)
            if 'connet_to_pt_tail' in eval_methods:
                #直接连接到pt_tail
                log_str=f'connet-to-pt-tail attack on model {simple_name} on attack method {method_prefix} and dataset {ds_args.dataset_name}'
                logger.info(log_str)
                print(log_str)
                loss,ppl,rouge_l_f,meteor=connet_to_pt_tail_evaluate(env_args=env_args,head_model=head_model,
                                           server_model=server_model,pt_tail_model=forzen_tail_model,
                                           tokenizer=tokenizer,data_loader=valid_data_loader)
                log_str=f"connet-to-pt-tail attack on model {simple_name} on attack method {method_prefix} with dataset {ds_args.dataset_name} ->\
                    loss: {loss:.4f} ppl: {ppl:.4f} rouge-l f1: {rouge_l_f:.4f} meteor: {meteor:.4f}\n"
                logger.info(log_str)
                print(log_str)
                torch.cuda.empty_cache()
        except Exception as e:
            err=f"Failed to eval model {simple_name} on attack method {method_prefix} and dataset {ds_args.dataset_name} error: {e},continue to next model"
            logger.info(err)
            print(err)
            continue
    #加载数据集
    #开始攻击
    pass


