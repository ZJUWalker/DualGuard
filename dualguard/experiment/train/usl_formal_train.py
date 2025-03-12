import os
import sys

sys.path.append(os.path.abspath('/home/wyz/deeplearning/workspace/DualGuard'))

import itertools
import math

from dualguard.defense.dp_config import DPConfig
from dualguard.utils.configs import DatasetArgs, EnvArgs, LogArgs, USLTrainArgs
from dualguard.utils.exp import AverageMeter, load_datasets
from dualguard.utils.model import calculate_meteor, calculate_rouge_text, decode_with_extra_space_evaluate, load_model_and_tokenizer, set_random_seed
from dualguard.usl import *
from dualguard.utils.logger import create_logger
from dualguard.experiment.method_config import *
from dualguard.utils import env
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy

from typing import Union
from transformers import AutoTokenizer
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers.models.llama import LlamaForCausalLM
from transformers.models.qwen2 import Qwen2ForCausalLM
from peft import get_peft_model, PeftModelForCausalLM

HeadModel=Union[QwenHead,LlamaHead,GPT2Head]
TailModel=Union[QwenTail,LlamaTail,GPT2Tail]
ServerModel=Union[QwenServer,LlamaServer,GPT2Server]

def _check_grad_and_params(model:nn.Module,layer_name):
    length=10
    for name, param in model.named_parameters():
        if param.requires_grad:
            if layer_name in name:
                print(f'{name},grad:{param.grad[0,:length].detach().cpu().numpy() if param.grad is not None else "no grad"},param:{param[0,:length].detach().cpu().numpy()}')
                break
            
def _frozen_model(split_model:Union[LlamaSplitModel,QwenSplitModel,GPT2SplitModel],config:USLTrainArgs):
    if config.frozen_head:
        for _, param in split_model.head_model.named_parameters():
            param.requires_grad = False
    if config.frozen_server:
        for _, param in split_model.server_model.named_parameters():
            param.requires_grad = False
    if config.frozen_tail:
        for _, param in split_model.tail_model.named_parameters():
            param.requires_grad = False
            
def _load_lora_model(split_model:Union[LlamaSplitModel,QwenSplitModel,GPT2SplitModel],config:USLTrainArgs):
    if config.use_lora:
        if not config.frozen_head:
            split_model.head_model=get_peft_model(split_model.head_model,config.lora_config)
        if not config.frozen_server:
            split_model.server_model=get_peft_model(split_model.server_model,config.lora_config)
        if not config.frozen_tail:
            split_model.tail_model=get_peft_model(split_model.tail_model,config.lora_config)
            
def _init_optimizer(model:nn.Module,usl_args:USLTrainArgs):
    lr=usl_args.optimizer_kwargs['lr']
    weight_decay=usl_args.optimizer_kwargs['weight_decay']
    optim=None
    params=list(filter(lambda p: p.requires_grad, model.parameters()))
    if len(params)>0:
        # print(params)
        optim=usl_args.optimizer(params,lr=lr,weight_decay=weight_decay)
    return optim


def usl_train(
    usl_args:USLTrainArgs, 
    env_args:EnvArgs, 
    head_model:HeadModel, 
    server_model:ServerModel, 
    tail_model:TailModel, 
    pt_tail_model:TailModel,
    tokenizer:AutoTokenizer,
    train_loader:DataLoader, 
    test_loader:DataLoader, 
    **kwargs
    ):
    # device = env_args.device
    #设置优化器
    head_optimizer=_init_optimizer(head_model,usl_args)
    server_optimizer=_init_optimizer(server_model,usl_args)
    tail_optimizer=_init_optimizer(tail_model,usl_args)
    # print(f"optimizer,memory{torch.cuda.memory_allocated(device)/1024/1024/1024:.3f}G")
    #开始·训练
    train_step=0
    best_metric=100000000
    no_improvement_epochs=0
    for epoch in itertools.count(start=1):
        train_step, no_improvement_epochs, best_metric, head_model, server_model, tail_model = train_validate_usl(
            usl_args=usl_args,
            env_args=env_args,
            head_model=head_model,
            server_model=server_model,
            tail_model=tail_model,
            pt_tail_model=pt_tail_model,
            optimizer_head=head_optimizer,
            optimizer_server=server_optimizer,
            optimizer_tail=tail_optimizer,
            # scheduler_server=scheduler_server,
            train_loader=train_loader,
            test_loader=test_loader,
            tokenizer=tokenizer,
            epoch=epoch,
            train_step=train_step,
            best_metric=best_metric,
            no_improvement_epochs=no_improvement_epochs,
            **kwargs
        )
        if no_improvement_epochs >= usl_args.early_stop_patience:
            print("-" * 100)
            print("连续 5 个 epoch 验证集损失无显著变化，停止训练。")
            # print("End of training")
            break
    return head_model, server_model, tail_model
    pass


def train_validate_usl(
    usl_args:USLTrainArgs,
    env_args:EnvArgs,
    head_model:HeadModel,
    server_model:ServerModel,
    tail_model:TailModel,
    pt_tail_model:TailModel,
    optimizer_head:torch.optim.Optimizer,
    optimizer_server:torch.optim.Optimizer,
    optimizer_tail:torch.optim.Optimizer,
    train_loader:DataLoader,
    test_loader:DataLoader,
    tokenizer:AutoTokenizer,
    epoch=1,
    train_step=0,
    no_improvement_epochs=0,
    best_metric=100000000,
    **kwargs
):
    threshold=usl_args.early_stop_threshold
    eval_interval=len(train_loader)//usl_args.validation_an_epoch
    # eval_interval=1 #测试代码
    device=env_args.device
    head_model.train()
    server_model.train()
    tail_model.train()
    avg_lm_loss = AverageMeter()#计算平均语言模型损失
    # Meter to average language model loss
    best_val_ppl = None
    log_list = []
    for idx, batch in enumerate(train_loader):
        train_step += 1
        _input = batch["input"].to(device)
        _target = batch["input"].to(device)
        _attention_mask = batch["attention_mask"].to(device)
        _msk = batch["_mask"].to(device) if "_mask" in batch.keys() else None
        # with autocast():/
        if isinstance(head_model, GPT2Head) or (isinstance(head_model, PeftModelForCausalLM) and isinstance(head_model.base_model.model, GPT2Head)):
            head_outputs=head_model(
                input_ids=_input,
                attention_mask=_attention_mask,
                use_cache=False,
            )
            hidden_states,presents,past_key_values,attention_mask,head_mask,encoder_hidden_states,encoder_attention_mask,use_cache,\
            output_attentions,output_hidden_states,all_self_attentions,all_hidden_states,all_cross_attentions=head_outputs
            hidden_states_from_head=hidden_states.clone().detach().requires_grad_(True)
            server_outputs=server_model(hidden_status_from_head=hidden_states_from_head,
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
                                                    all_cross_attentions=all_cross_attentions,
                                                    all_hidden_states=all_hidden_states)
            hidden_states_from_server,presents,past_key_values,attention_mask,head_mask,encoder_hidden_states,encoder_attention_mask,use_cache,\
            output_attentions,output_hidden_states,all_self_attentions,all_hidden_states,all_cross_attentions=server_outputs
            tail_outputs=tail_model(hidden_status_from_server=hidden_states_from_server,
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
                                                    lm_mask=_msk)
            lm_loss=tail_outputs.loss
            pt_lm_loss=0
            if pt_tail_model is not None:
                pt_tail_outputs=pt_tail_model(hidden_status_from_server=hidden_states_from_server,
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
                                    lm_mask=_msk)
                pt_lm_loss=pt_tail_outputs.loss
                total_loss = lm_loss + usl_args.lambda_3 / pt_lm_loss
            else:
                total_loss = lm_loss
            pass
        # elif isinstance(head_model, (LlamaHead,QwenHead)):
        else:
            head_outputs = head_model(input_ids=_input,attention_mask=_attention_mask,use_cache=False)
            hidden_states_from_head = head_outputs[0].clone().detach().requires_grad_(True)#冻结头部模型
            server_outputs = server_model(
                    hidden_status_from_head=hidden_states_from_head,
                    attention_mask=head_outputs[1],
                    position_ids=head_outputs[2],
                    past_key_values=head_outputs[3],
                    output_attentions=head_outputs[4],
                    use_cache=head_outputs[5],
                    cache_position=head_outputs[6],
                    all_hidden_states=head_outputs[7],
                    all_self_attns=head_outputs[8],
                    return_legacy_cache=head_outputs[9],
                )

                # print(f'server output: {temp_outputs[0]}')
            tail_outputs = tail_model(
                hidden_status_from_server=server_outputs[0],
                attention_mask=server_outputs[1],
                position_ids=server_outputs[2],
                past_key_values=server_outputs[3],
                output_attentions=server_outputs[4],
                use_cache=server_outputs[5],
                cache_position=server_outputs[6],
                all_hidden_states=server_outputs[7],
                all_self_attns=server_outputs[8],
                return_legacy_cache=server_outputs[9],
                labels=_target,
                lm_mask=_msk,
            )

            lm_loss=tail_outputs.loss
            pt_lm_loss=0
            if pt_tail_model is not None:

                pt_tail_outputs = pt_tail_model(
                    hidden_status_from_server=server_outputs[0],
                    attention_mask=server_outputs[1],
                    position_ids=server_outputs[2],
                    past_key_values=None,
                    output_attentions=server_outputs[4],
                    use_cache=False,
                    cache_position=server_outputs[6],
                    all_hidden_states=server_outputs[7],
                    all_self_attns=server_outputs[8],
                    return_legacy_cache=server_outputs[9],
                    labels=_target,
                    lm_mask=_msk,
                )
                pt_lm_loss=pt_tail_outputs.loss
                total_loss = lm_loss + usl_args.lambda_3 / pt_lm_loss
            else:
                total_loss = lm_loss
            pass
        avg_lm_loss.update(lm_loss.item())
        #反向传播
        total_loss.backward()
        torch.cuda.empty_cache()
        #优化器更新
        if not usl_args.frozen_tail:
            optimizer_tail.step()
            optimizer_tail.zero_grad()
        if not usl_args.frozen_server:
            optimizer_server.step()
            optimizer_server.zero_grad()
        if not usl_args.frozen_head:
            head_outputs[0].backward(hidden_states_from_head.grad)
            optimizer_head.step()
            optimizer_head.zero_grad()
        if train_step % usl_args.log_interval == 0:
            if pt_tail_model is not None:
                log_str = (
                    f"| epoch {epoch:3d} step {train_step:>8d} |"
                    f"loss {avg_lm_loss.val:5.2f} | avg loss {avg_lm_loss.avg:5.2f} | "
                    f"ppl {math.exp(avg_lm_loss.avg):5.2f} | _lm_loss {lm_loss.item():5.2f} | _lm_loss_before {pt_lm_loss.item():5.2f} | total_loss {total_loss.item():5.2f} |"
                )
            else:
                log_str = (
                    f"| epoch {epoch:3d} step {train_step:>8d} |"
                    f"loss {avg_lm_loss.val:5.2f} | avg loss {avg_lm_loss.avg:5.2f} | "
                    f"ppl {math.exp(avg_lm_loss.avg):5.2f} | _lm_loss {lm_loss.item():5.2f} | total_loss {total_loss.item():5.2f} |"
                )
            logger.info(log_str)
            log_list.append(log_str)
            print(log_str)
            avg_lm_loss.reset()

        if train_step % eval_interval== 0:   
            valid_loss, valid_ppl = evaluate_usl(
                usl_args=usl_args,
                env_args=env_args,
                head_model=head_model,
                server_model=server_model,
                tail_model=tail_model,
                valid_loader=test_loader,
                tokenizer=tokenizer,
                output_similarity=False
            )

            if best_val_ppl is None or valid_ppl < best_val_ppl:
                best_val_ppl = valid_ppl

            log_str = (
                f"| Eval {train_step // eval_interval:3d} at step {train_step:>8d} | "
                f"valid ppl {valid_ppl:5.2f} | best ppl {best_val_ppl:5.2f} "
                f"valid loss {valid_loss:5.2f} | best loss {best_metric:5.2f} |"
            )
            log_list.append(log_str)
            logger.info(log_str)
            print("-" * 100)
            print(log_str)
            print("-" * 100)

            head_model.train()
            server_model.train()
            tail_model.train()

            print(valid_loss, best_metric, threshold)
            if valid_loss < best_metric:
                if valid_loss < best_metric - threshold:
                    no_improvement_epochs = -1
                best_metric = valid_loss
            no_improvement_epochs += 1
            # print(no_improvement_epochs, best_metric)
            print(no_improvement_epochs, best_metric)
            if no_improvement_epochs >= 5:
                return train_step, no_improvement_epochs, best_metric, head_model, server_model, tail_model
        torch.cuda.empty_cache()
    return train_step, no_improvement_epochs, best_metric, head_model, server_model, tail_model

def evaluate_usl(
    usl_args:USLTrainArgs,
    env_args:EnvArgs,
    head_model:HeadModel,
    server_model:ServerModel, 
    tail_model:TailModel, 
    valid_loader:DataLoader, 
    tokenizer:AutoTokenizer, 
    output_similarity=False
    ):
    head_model.eval()
    server_model.eval()
    tail_model.eval()

    device = env_args.device
    # head_model.to(device)
    # server_model.to(device)
    # tail_model.to(device)
    avg_lm_loss = AverageMeter()
    rouge_l_f = 0.0
    meteor = 0.0
    with torch.no_grad():
        for idx, batch in enumerate(valid_loader):
            _input = batch["input"].to(device)
            _target = batch["input"].to(device)
            _attention_mask = batch["attention_mask"].to(device)
            _msk = batch["_mask"].to(device)
            if isinstance(head_model, GPT2Head) or (isinstance(head_model, PeftModelForCausalLM) and isinstance(head_model.base_model.model, GPT2Head)):
                temp=head_model(
                    input_ids=_input,
                    attention_mask=_attention_mask
                )
                hidden_states,presents,past_key_values,attention_mask,head_mask,encoder_hidden_states,encoder_attention_mask,use_cache,\
                output_attentions,output_hidden_states,all_self_attentions,all_hidden_states,all_cross_attentions=temp
                temp=server_model(hidden_status_from_head=hidden_states,
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
                                                        all_cross_attentions=all_cross_attentions,
                                                        all_hidden_states=all_hidden_states)
                hidden_states,presents,past_key_values,attention_mask,head_mask,encoder_hidden_states,encoder_attention_mask,use_cache,\
                output_attentions,output_hidden_states,all_self_attentions,all_hidden_states,all_cross_attentions=temp
                tail_outputs=tail_model(hidden_status_from_server=hidden_states,
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
                                                        lm_mask=_msk)
                _loss=tail_outputs.loss
                _logits=tail_outputs.logits
            else:
                head_outputs = head_model(input_ids=_input,attention_mask=_attention_mask)
                hidden_states_from_head = head_outputs[0].clone().detach().requires_grad_(True)#冻结头部模型
                server_outputs = server_model(
                        hidden_status_from_head=hidden_states_from_head,
                        attention_mask=head_outputs[1],
                        position_ids=head_outputs[2],
                        past_key_values=None,
                        output_attentions=head_outputs[4],
                        use_cache=False,
                        cache_position=head_outputs[6],
                        all_hidden_states=head_outputs[7],
                        all_self_attns=head_outputs[8],
                        return_legacy_cache=head_outputs[9],
                    )
                    # print(f'server output: {temp_outputs[0]}')
                tail_outputs = tail_model(
                    hidden_status_from_server=server_outputs[0],
                    attention_mask=server_outputs[1],
                    position_ids=server_outputs[2],
                    past_key_values=None,
                    output_attentions=server_outputs[4],
                    use_cache=False,
                    cache_position=server_outputs[6],
                    all_hidden_states=server_outputs[7],
                    all_self_attns=server_outputs[8],
                    return_legacy_cache=server_outputs[9],
                    labels=_target,
                    lm_mask=_msk,
                )
                _loss=tail_outputs.loss
                _logits=tail_outputs.logits
            # else:
            #     raise ValueError("Unsupported head model type")
            
            avg_lm_loss.update(_loss.item())
            if idx % usl_args.log_interval == 0:
                print("eval samples:", idx, "loss:", _loss.item())
            
            # _loss = calc_unshift_loss(_logits, _target, _msk, args.label_smooth)
            if output_similarity:
                shift_logits = _logits[..., :-1, :].contiguous()  # 去掉最后一个时间步
                shift_labels = _target[..., 1:].contiguous()     # 去掉第一个时间步
                shift_mask = _msk[..., 1:].contiguous()
                atk_txts = [decode_with_extra_space_evaluate(tokenizer, s, m) for s, m in zip(shift_logits.argmax(dim=-1), shift_mask)]
                gt_txts = [decode_with_extra_space_evaluate(tokenizer, s, m) for s, m in zip(shift_labels, shift_mask)]
                rouge = calculate_rouge_text(atk_txts, gt_txts, print_comparison=False)
                _meteor = calculate_meteor(atk_txts, gt_txts)
                rouge_l_f += rouge["rouge-l"]["f"]
                meteor +=_meteor
                
        print("average loss", avg_lm_loss.avg)
    if output_similarity:
        return avg_lm_loss.avg, math.exp(avg_lm_loss.avg), rouge_l_f/len(valid_loader), meteor/len(valid_loader)
    else:
        return avg_lm_loss.avg, math.exp(avg_lm_loss.avg)
    

def _load_warmup_model_and_tokenizer(usl_args:USLTrainArgs,dataset_args:DatasetArgs,env_args:EnvArgs,dp_config:DPConfig=None):
    simple_name=usl_args.model_name.split('/')[-1]
    add_lora=usl_args.use_lora
    lora_config=usl_args.lora_config
    pt_model,tokenizer=load_model_and_tokenizer(usl_args.model_name) #warmup训练不需要server
    split_model:Union[LlamaSplitModel,QwenSplitModel,GPT2SplitModel]=None
    logger.info(f'Loading {simple_name} model and tokenizer...')
    #拆分模型
    if isinstance(pt_model, GPT2LMHeadModel):
        split_model=GPT2SplitModel(pt_model,SplitModelConfig(usl_args.split_point,-1,usl_args.split_point,True,True),dp_config=dp_config)
    elif isinstance(pt_model, LlamaForCausalLM):
        split_model=LlamaSplitModel(pt_model,SplitModelConfig(usl_args.split_point,-1,usl_args.split_point,True,True),dp_config=dp_config)
    elif isinstance(pt_model, Qwen2ForCausalLM):
        split_model=QwenSplitModel(pt_model,SplitModelConfig(usl_args.split_point,-1,usl_args.split_point,True,True),dp_config=dp_config)
    else:
        raise ValueError("Unsupported model type")
    if dp_config is None:
        split_model.disable_dp() #不需要加噪声
    else:
        split_model.enable_dp() #需要加噪声
    #加载pt_tail_model（如果需要）
    pt_tail_model=None
    #是否需要冗余的pt_tail_model
    if usl_args.with_pt_tail_model:
        pt_tail_model=copy.deepcopy(split_model.tail_model) #pt_tail_model用于usl训练中的pt_lm_loss，不需要lora
        for _, param in pt_tail_model.named_parameters():
            param.requires_grad = False
    if not usl_args.use_naive_usl:
        #从warmup模型中加载参数
        wp_model_dir=usl_args.wp_model_dir
        split_model.head_model=torch.load(os.path.join(wp_model_dir,f'{simple_name}/{dataset_args.dataset_name}/head.pth'),map_location=env_args.device,weights_only=False)
        split_model.tail_model=torch.load(os.path.join(wp_model_dir,f'{simple_name}/{dataset_args.dataset_name}/tail.pth'),map_location=env_args.device,weights_only=False)
        # split_model.tail_model=torch.load(os.path.join(wp_model_dir,f'{simple_name}/tail_{dataset_args.dataset_name}.pth'),map_location=env_args.device)
    torch.cuda.empty_cache()
    #冻结不需要的模型
    _frozen_model(split_model,usl_args)
    #对需要加载lora的模型加载lora
    _load_lora_model(split_model,usl_args)
    #打印可微调的模型参数数量
    _print_trainable_parameters(split_model.head_model)
    _print_trainable_parameters(split_model.server_model)
    _print_trainable_parameters(split_model.tail_model)
    #加载到device
    split_model.head_model.to(env_args.device)
    split_model.server_model.to(env_args.device)
    split_model.tail_model.to(env_args.device)
    if pt_tail_model is not None:
        pt_tail_model.to(env_args.device)
    return split_model,pt_tail_model,tokenizer

def _print_trainable_parameters(model:nn.Module):
    trainable_params_count=sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_prams_count=sum(p.numel() for p in model.parameters())
    print(f'trainable params count:{trainable_params_count}, total params count:{total_prams_count},ratio:{trainable_params_count/total_prams_count:.3f}')    


total_args=[dualguard_config]#可拔插的训练参数

if __name__ == '__main__':
    #默认值
    ds_args = DatasetArgs(dataset_name=GSM8K)
    env_args = EnvArgs(device='cuda:1')
    set_random_seed(env_args.random_seed)
    for args in total_args:
        usl_args=args.usl_args
        usl_args.log_interval=100
        dp_config=args.dp_config
        usl_args.model_name=GPT2_LARGE
        save_prefix=args.prefix
        simple_name=usl_args.model_name.split('/')[-1]#用于日志和保存模型
        if usl_args.use_lora:
            usl_args.lora_config=gpt_lora_config
        # 配置日志记录
        logger=create_logger(LogArgs(
            log_dir=env.usl_log_dir,
            log_file_name=f'{simple_name}_{args.prefix}_{ds_args.dataset_name}.log'
        ))
        if dp_config.add_noise:
            log_str=f"\n{'='*20} {args.info} on model {simple_name} dataset {ds_args.dataset_name} with dp epsilon: {dp_config.epsilon} {'='*20}\n"
        else:
            log_str=f"\n{'='*20} {args.info} on model {simple_name} dataset {ds_args.dataset_name} without dp {'='*20}\n"
        logger.info(log_str)
        print(log_str)
        split_model,pt_tail_model,tokenizer=_load_warmup_model_and_tokenizer(usl_args,ds_args,env_args,dp_config)
        head_model,server_model,tail_model=split_model.head_model,split_model.server_model,split_model.tail_model
        # print(head_model)
        # print(f'memory used: {torch.cuda.memory_allocated(device=env_args.device)/1024/1024/1024:.3f}GB')
        #加载数据集
        data_loaders=load_datasets(ds_args,tokenizer=tokenizer)
        train_data_loader=data_loaders['train']
        valid_data_loader=data_loaders['validation'] if 'validation' in data_loaders.keys() else data_loaders['test']
        '''
        USL train
        1.常规训练+pretrain model loss
        2.冻结头部模型
        3.训练+验证
        4.保存模型，三个模型都保存'''
        # usl_head_model, usl_server_model, usl_tail_model=head_model,server_model,tail_model #测试
        usl_head_model, usl_server_model, usl_tail_model = usl_train(
            usl_args=usl_args,
            env_args=env_args,
            head_model=head_model,
            server_model=server_model,
            tail_model=tail_model,
            pt_tail_model=pt_tail_model,
            tokenizer=tokenizer,
            train_loader=train_data_loader,
            test_loader=valid_data_loader,
        )
        # # 验证
        avg_loss, ppl, rouge_l_f, meteor = evaluate_usl(usl_args,env_args,head_model,server_model,tail_model,valid_data_loader,tokenizer,output_similarity=True)
        res_str=f"evaluate_usl on model {simple_name},dataset {ds_args.dataset_name} and method {save_prefix},\
        eps: {dp_config.epsilon if dp_config.add_noise else False}, -> average loss: {avg_loss}, ppl: {ppl}, rouge-l-f: {rouge_l_f}, meteor: {meteor}"
        logger.info(res_str)
        print(res_str)
        # save models
        save_dir=os.path.join(usl_args.save_dir,simple_name,ds_args.dataset_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # save_dir=f"{usl_args.save_dir}{simple_name}/{ds_args.dataset_name}/{save_prefix}"
        if save_prefix not in ['dualguard','naive_usl']:
            save_prefix+=f"_e_{dp_config.epsilon}"
        # print(f"Saving models to {save_dir}...")
        if not usl_args.frozen_head:
            torch.save(usl_head_model,f"{save_dir}/{save_prefix}_head.pth")
        torch.save(usl_server_model,f"{save_dir}/{save_prefix}_server.pth")
        torch.save(usl_tail_model,f"{save_dir}/{save_prefix}_tail.pth")
        logger.info(f"Saved models to {save_dir}")
        del head_model,server_model,tail_model,pt_tail_model,tokenizer
        del usl_head_model,usl_server_model,usl_tail_model,train_data_loader,valid_data_loader,data_loaders
        del split_model
        torch.cuda.empty_cache()
        print(f'memory used after release: {torch.cuda.memory_allocated(device=env_args.device)/1024/1024/1024:.3f}GB')
        # break
    pass