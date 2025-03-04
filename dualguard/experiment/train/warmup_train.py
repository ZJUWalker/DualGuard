import os
import sys

sys.path.append(os.path.abspath('/home/wyz/deeplearning/workspace/Privacy-USL-LLM'))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
import logging
import logging.handlers

from typing import Union,Tuple
from transformers import AutoTokenizer
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers.models.llama import LlamaForCausalLM
from transformers.models.qwen2 import Qwen2ForCausalLM
from peft import get_peft_model, LoraConfig, TaskType,PeftModelForCausalLM

from dualguard.usl import *
from dualguard.utils.configs import DatasetArgs, EnvArgs, LogArgs, WarmupArgs
from dualguard.utils.exp import load_datasets
from dualguard.utils.model import calc_unshift_loss, load_model_and_tokenizer,get_embed_size,get_vocab_size,set_random_seed
from dualguard.utils.logger import create_logger
from dualguard.attack.sip import load_sip_model,GRUDRInverter
# sfl.model.attacker.sip.inversion_models.GRUDRInverter
HeadModel=Union[QwenHead,LlamaHead,GPT2Head]
TailModel=Union[QwenTail,LlamaTail,GPT2Tail]
ServerModel=Union[QwenServer,LlamaServer,GPT2Server]

#投影层
class ProjectionMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        """
        :param hidden_size: 输入和输出的特征维度大小
        :param intermediate_size: 中间隐藏层的特征维度大小
        """
        super(ProjectionMLP, self).__init__()
        
        # 第一层：从 hidden_size 到 intermediate_size
        self.layer1 = nn.Linear(hidden_size, intermediate_size)
        self.layer2 = nn.Linear(intermediate_size, intermediate_size)
        self.layer3 = nn.Linear(intermediate_size, hidden_size)  # 输出回到 hidden_size

        # 激活函数
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        :param x: 输入张量，形状为 (batch_size, sequence_length, hidden_size)
        :return: 输出张量，形状为 (batch_size, sequence_length, hidden_size)
        """
        # 第一层 + 激活
        x = self.activation(self.layer1(x))
        
        # 第二层 + 激活
        x = self.activation(self.layer2(x))
        
        # 第三层（无激活）
        x = self.layer3(x)
        
        return x


def warmup_validation(
    wramup_args:WarmupArgs,
    head:HeadModel,
    proj:nn.Module, 
    tail:TailModel, 
    pt_tail_model:TailModel, 
    attack_model:nn.Module, 
    valid_loader:DataLoader):
    head.eval()
    proj.eval()
    tail.eval()
    attack_model.eval()
    pt_tail_model.eval()
        # avg_lm_loss = AverageMeter()
    _lm_losses=[]
    _attack_losses=[]
    _total_losses = []
    _lm_losses_before = []
    device=env_args.device
    with torch.no_grad():
        for idx, data in enumerate(valid_loader):
            _input = data["input"].to(device)
            _target = data["input"].to(device)
            _mask = data["attention_mask"].to(device)
            _lm_mask=data["_mask"].to(device) if data.get("_mask") is not None else None
            head_output=head.forward(input_ids=_input,attention_mask=_mask)
            # hidden_states_from_head
            tail_output={}
            #获取warmup的中间结果和loss 
            if isinstance(head, (LlamaHead,QwenHead)) or (isinstance(head, PeftModelForCausalLM) and isinstance(head.base_model.model, (LlamaHead,QwenHead))):
                hidden_states_from_head,causal_mask,position_ids,\
                past_key_values,output_attentions,use_cache,cache_position,\
                all_hidden_states,all_self_attns,return_legacy_cache=head_output
                project_hidden_states=proj(hidden_states_from_head)
                tail_output=tail.forward(
                    hidden_status_from_server=project_hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    all_hidden_states=all_hidden_states,
                    all_self_attns=all_self_attns,
                    return_legacy_cache=return_legacy_cache,
                    labels=_target,
                    lm_mask=_lm_mask
                )
                pass
            else:
                hidden_states_from_head,presents,past_key_values,attention_mask,head_mask,\
                encoder_hidden_states,encoder_attention_mask,use_cache,\
                output_attentions,output_hidden_states,all_self_attentions,all_hidden_states,all_cross_attentions=head_output
                project_hidden_states=proj(hidden_states_from_head)
                tail_output=tail.forward(
                    hidden_status_from_server=project_hidden_states,
                    presents=presents,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    all_self_attentions=all_self_attentions,
                    all_hidden_states=all_hidden_states,
                    all_cross_attentions=all_cross_attentions,
                    labels=_target,
                    lm_mask=_lm_mask
                )
            lm_loss=tail_output.loss
            #获取经过sip模型的输出loss
            attack_logits=attack_model(hidden_states_from_head)
            if isinstance(tail, (LlamaTail,QwenTail)) or (isinstance(tail, PeftModelForCausalLM) and isinstance(tail.base_model.model, (LlamaTail,QwenTail))):
                pt_tail_output=pt_tail_model.forward(
                    hidden_status_from_server=project_hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    output_attentions=output_attentions,
                    use_cache=False,
                    cache_position=cache_position,
                    all_hidden_states=all_hidden_states,
                    all_self_attns=all_self_attns,
                    return_legacy_cache=return_legacy_cache,
                    labels=_target,
                    lm_mask=_lm_mask
                )
            else:
                pt_tail_output=pt_tail_model.forward(
                    hidden_status_from_server=project_hidden_states,
                    presents=presents,
                    past_key_values=None,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=False,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    all_self_attentions=all_self_attentions,
                    all_hidden_states=all_hidden_states,
                    all_cross_attentions=all_cross_attentions,
                    labels=_target,  
                    lm_mask=_lm_mask       
                )    
            pt_tail_loss=pt_tail_output.loss
            
            _attack_loss = calc_unshift_loss(attack_logits, _input)
            total_loss = wp_args.lambda_1 /_attack_loss + lm_loss + wp_args.lambda_2 / pt_tail_loss

            _total_losses.append(total_loss.item())
            _lm_losses.append(lm_loss.item())
            _attack_losses.append(_attack_loss.item())
            _lm_losses_before.append(pt_tail_loss.item())
            if idx % 100 == 0:
                print("eval samples:", idx, "loss:", sum(_total_losses) / len(_total_losses))

        return sum(_total_losses) / len(_total_losses), sum(_lm_losses) / len(_lm_losses), \
                sum(_attack_losses) / len(_attack_losses), sum(_lm_losses_before) / len(_lm_losses_before)
    
def warmup_train(wp_args:WarmupArgs,
                 env_args:EnvArgs,
                 head_model:HeadModel,
                 tail_model:TailModel,
                 pt_tail_model:TailModel,#冻结的预训练尾部模型
                 attack_model:nn.Module,
                 train_loader:DataLoader,
                 valid_loader:DataLoader,
                 tokenizer:AutoTokenizer,
                 **kwargs
                 ):
    device=env_args.device
    logger=kwargs.get("logger") if kwargs.get("logger") is not None else logging.getLogger(__name__)
    #设置优化器
    head_optimizer=wp_args.optimizer(head_model.parameters(),
                                         lr=wp_args.optimizer_kwargs['lr'],
                                         weight_decay=wp_args.optimizer_kwargs['weight_decay'])
    tail_optimizer=wp_args.optimizer(tail_model.parameters(),
                                         lr=wp_args.optimizer_kwargs['lr'],
                                         weight_decay=wp_args.optimizer_kwargs['weight_decay'])
    #设置辅助模型
    hidden_size=0
    if isinstance(head_model, (LlamaHead,QwenHead)) or (isinstance(head_model, PeftModelForCausalLM) and isinstance(head_model.base_model.model, (LlamaHead,QwenHead))):
        hidden_size=head_model.config.hidden_size
    else:
        hidden_size=head_model.config.n_embd
    project_mlp=ProjectionMLP(hidden_size=hidden_size,intermediate_size=hidden_size*2).to(device)
    pmlp_optimizer=wp_args.warm_up_optimizer(project_mlp.parameters(),
                                                 lr=wp_args.warm_up_optimizer_kwargs['lr'],
                                                 weight_decay=wp_args.warm_up_optimizer_kwargs['weight_decay'])
    head_model.train()
    tail_model.train()
    project_mlp.train()
    attack_model.train()
    pt_tail_model.train()
    validation_interval=len(train_loader)//wp_args.validation_an_epoch
    #开始训练
    for epoch in range(wp_args.warm_up_epochs):
        total_loss=0
        total_attack_loss=0
        total_pt_tail_loss=0
        total_lm_loss=0
        # print(head_model.wte.weight.grad is None)

        for idx, batch in enumerate(train_loader):
            head_optimizer.zero_grad()
            tail_optimizer.zero_grad()
            pmlp_optimizer.zero_grad()
            # #前向传播得到三部分loss
            lm_loss,attack_loss,pt_tail_loss=_warmup_forward(head_model, project_mlp,tail_model,pt_tail_model,attack_model,batch,device)
            total_attack_loss+=attack_loss.item()
            
            total_pt_tail_loss+=pt_tail_loss.item()
            total_lm_loss+=lm_loss.item()
            # #合并loss
            bwd_loss=wp_args.lambda_1 /attack_loss + lm_loss + wp_args.lambda_2 / pt_tail_loss
            total_loss+=bwd_loss.item()
            #反向传播
            bwd_loss.backward()
            # if head_model.wte.weight.grad is not None:
            #     print(head_model.wte.weight.grad[:,10])
            #更新参数
            head_optimizer.step()
            tail_optimizer.step()
            pmlp_optimizer.step()
            tail_model.zero_grad()

            torch.cuda.empty_cache()
            if (idx+1) % wp_args.log_interval == 0:
                log_str = (
                    f"| epoch: {epoch+1} step: {idx+1}"
                    f"| lm_loss: {total_lm_loss /wp_args.log_interval} | attack_loss: {total_attack_loss / wp_args.log_interval} "
                    f"| lm_loss_before: {total_pt_tail_loss / wp_args.log_interval} | totol_loss: {total_loss / wp_args.log_interval}"
                    f"| memory: {torch.cuda.memory_allocated(device) / 1024 ** 3:.3f}GB"
                )
                logger.info(log_str)
                print(log_str)
                total_loss=0
                total_attack_loss=0
                total_pt_tail_loss=0
                total_lm_loss=0
            if (idx+1) % validation_interval == 0:
                valid_total_loss, valid_lm_loss, valid_attack_loss, valid_lm_loss_before=warmup_validation(wp_args,head_model,project_mlp,tail_model,pt_tail_model,attack_model,valid_loader)
                log_str = (
                    f"| Eval {(idx+1) // validation_interval:3d} at step {(idx+1):>8d} | "
                    f"valid lm_loss {valid_lm_loss:5.2f} | "
                    f"valid attack_loss {valid_attack_loss:5.2f} | valid_lm_loss_before {valid_lm_loss_before:5.2f} | valid total loss {valid_total_loss:5.2f}"
                )
                logger.info(log_str)
                print(log_str)
                head_model.train()
                tail_model.train()
                project_mlp.train()
                attack_model.train()
                pt_tail_model.train()
    return head_model, tail_model
    pass

def _warmup_forward(head_model:HeadModel,
                    project_mlp:nn.Module,
                    tail_model:TailModel,
                    pt_tail_model:TailModel,
                    attack_model:nn.Module,
                    batch:dict,
                    device:torch.device):
    _input = batch["input"].to(device)
    _target = batch["input"].to(device)
    _mask = batch["_mask"].to(device) if "_mask" in batch.keys() else None
    _atten_mask=batch["attention_mask"].to(device)
    head_output=head_model.forward(input_ids=_input,attention_mask=_atten_mask)
    # hidden_states_from_head
    #获取warmup的中间结果和loss 
    if isinstance(tail_model, (LlamaTail,QwenTail)) or (isinstance(tail_model, PeftModelForCausalLM) and isinstance(tail_model.base_model.model, (LlamaTail,QwenTail))):
        hidden_states_from_head,causal_mask,position_ids,\
        past_key_values,output_attentions,use_cache,cache_position,\
        all_hidden_states,all_self_attns,return_legacy_cache=head_output
        project_hidden_states=project_mlp(hidden_states_from_head)
        tail_output=tail_model.forward(
            hidden_status_from_server=project_hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            all_hidden_states=all_hidden_states,
            all_self_attns=all_self_attns,
            return_legacy_cache=return_legacy_cache,
            labels=_target,
            lm_mask=_mask
        )
        pass
    else:
        hidden_states_from_head,presents,past_key_values,attention_mask,head_mask,\
        encoder_hidden_states,encoder_attention_mask,use_cache,\
        output_attentions,output_hidden_states,all_self_attentions,all_hidden_states,all_cross_attentions=head_output
        project_hidden_states=project_mlp(hidden_states_from_head)
        tail_output=tail_model.forward(
            hidden_status_from_server=project_hidden_states,
            presents=presents,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            all_self_attentions=all_self_attentions,
            all_hidden_states=all_hidden_states,
            all_cross_attentions=all_cross_attentions,
            labels=_target,
            lm_mask=_mask
        )
    lm_loss=tail_output.loss
    #获取经过sip模型的输出loss
    # detached_head_output_hs=hidden_states_from_head.clone().detach()
    # with torch.no_grad():
    # attack_logits=attack_model(detached_head_output_hs)
    attack_logits=attack_model(hidden_states_from_head)
    attack_loss = calc_unshift_loss(attack_logits, _input)
        # attack_loss=attack_output.loss
    #获取经过pretrained的tail_model的输出loss
    # detached_project_output_hs=project_hidden_states.clone().detach()
    # with torch.no_grad():
    if isinstance(tail_model, (LlamaTail,QwenTail)) or (isinstance(tail_model, PeftModelForCausalLM) and isinstance(tail_model.base_model.model, (LlamaTail,QwenTail))):
        pt_tail_output=pt_tail_model.forward(
            hidden_status_from_server=project_hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=None,
            output_attentions=output_attentions,
            use_cache=False,
            cache_position=cache_position,
            all_hidden_states=all_hidden_states,
            all_self_attns=all_self_attns,
            return_legacy_cache=return_legacy_cache,
            labels=_target,
            lm_mask=_mask
        )
    else:
        pt_tail_output=pt_tail_model.forward(
            hidden_status_from_server=project_hidden_states,
            presents=presents,
            past_key_values=None,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            all_self_attentions=all_self_attentions,
            all_hidden_states=all_hidden_states,
            all_cross_attentions=all_cross_attentions,
            labels=_target,       
            lm_mask=_mask
        )    
    pt_tail_loss=pt_tail_output.loss
    return lm_loss,attack_loss,pt_tail_loss

    pass


def _load_split_model_from_pretrained(warmup_args:WarmupArgs,split_config:SplitModelConfig)->Tuple[Union[LlamaSplitModel,QwenSplitModel,GPT2SplitModel],AutoTokenizer]:  
    pt_model,tokenizer=load_model_and_tokenizer(warmup_args.model_name)
    split_model:Union[LlamaSplitModel,QwenSplitModel,GPT2SplitModel]=None
    if isinstance(pt_model, GPT2LMHeadModel):
        split_model=GPT2SplitModel(pt_model,split_config)
    elif isinstance(pt_model, LlamaForCausalLM):
        split_model=LlamaSplitModel(pt_model,split_config)
    elif isinstance(pt_model, Qwen2ForCausalLM):
        split_model=QwenSplitModel(pt_model,split_config)
    else:
        raise ValueError("Unsupported model type")
    split_model.disable_dp() #不需要加噪声
    return split_model,tokenizer


def _load_models_and_tokenizer(wp_args:WarmupArgs,env_args:EnvArgs):
    model_name=wp_args.model_name
    simple_model_name=model_name.split('/')[-1]
    device=env_args.device
    #加载模型和tokenizer
    split_layer_num=2 if 'Llama' in model_name else 3
    split_model,tokenizer=_load_split_model_from_pretrained(wp_args,SplitModelConfig(split_layer_num,-1,split_layer_num,False,True)) #warmup训练不需要server
    split_model.disable_dp()#禁用加噪声
    #加载预训练的tail_model
    pt_tail_model=copy.deepcopy(split_model.tail_model)
    #加载attack model
    attack_model=torch.load(os.path.join(wp_args.sip_model_dir,f'sip_{simple_model_name}.pth'),weights_only=False)
    if wp_args.use_lora:
        split_model.head_model=get_peft_model(split_model.head_model,peft_config=wp_args.lora_config)
        split_model.tail_model=get_peft_model(split_model.tail_model,peft_config=wp_args.lora_config)
    #冻结模型参数
    for n,p in attack_model.named_parameters():
        p.requires_grad=False
    for n,p in pt_tail_model.named_parameters():
        p.requires_grad=False
    split_model.to(device)
    pt_tail_model.to(device)
    attack_model.to(device)
    return split_model,pt_tail_model,attack_model,tokenizer
    pass

if __name__ == '__main__':
    env_args = EnvArgs(device='cuda:0')
    set_random_seed(env_args.random_seed)
    wp_args = WarmupArgs(warm_up_epochs=4,validation_an_epoch=2,use_lora=True,log_interval=100,lora_config=LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=2,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["attn.c_proj", "attn.c_attn"]
        # target_modules=['q_proj','v_proj']
    ),model_name='gpt/gpt2-large')
    ds_args = DatasetArgs(dataset_name='gsm8k')
    #默认值
    simple_model_name=wp_args.model_name.split('/')[-1]#用于日志和保存模型    
    # 配置日志记录
    filename=f'warmup_training_{simple_model_name}_{ds_args.dataset_name}.log'  # 日志文件名
    # logger=create_logger(f'logger_{simple_model_name}_{dataset_args.dataset_name}',filename)
    logger=create_logger(log_args=LogArgs(log_dir='/home/wyz/deeplearning/workspace/Privacy-USL-LLM/dualguard/log/warmup',log_file_name=filename))
    logger.info(f"\n{'='*48} warmup training on model {simple_model_name} on dataset {ds_args.dataset_name} with lora config {wp_args.lora_config} {'='*48}")
    #加载模型和tokenizer
    split_model,pt_tail_model,attack_model,tokenizer=_load_models_and_tokenizer(wp_args,env_args)
    #加载数据集
    data_loaders=load_datasets(ds_args,tokenizer=tokenizer)
    train_data_loader=data_loaders['train']
    valid_data_loader=data_loaders['validation'] if 'validation' in data_loaders.keys() else data_loaders['test']
    #训练模型
    wramup_head_model,wramup_tail_model=warmup_train(
        wp_args=wp_args,
        env_args=env_args,
        head_model=split_model.head_model,
        tail_model=split_model.tail_model,
        pt_tail_model=pt_tail_model,
        attack_model=attack_model,
        train_loader=train_data_loader,
        valid_loader=valid_data_loader,
        tokenizer=tokenizer,
        logger=logger
        )
    logger.info(f"{'='*48} warmup training end {'='*48}")
    # 保存模型
    save_dir=os.path.join(wp_args.save_dir,f"{simple_model_name}",ds_args.dataset_name)
    if not os.path.exists(save_dir):
        logger.info(f"create save dir {save_dir}")
        os.makedirs(save_dir)
    head_ph=os.path.join(save_dir,"head.pth")
    tail_ph=os.path.join(save_dir,"tail.pth")
    logger.info(f"save head model to {head_ph}")
    torch.save(wramup_head_model,head_ph)
    logger.info(f"save tail model to {tail_ph}")
    torch.save(wramup_tail_model,tail_ph)