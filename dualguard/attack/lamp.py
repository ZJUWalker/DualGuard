# import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers.models.qwen2 import Qwen2ForCausalLM
from transformers.models.llama import LlamaForCausalLM


from dualguard.attack.tag import forward_and_get_true_grads
from dualguard.usl import *
from typing import Union

import numpy as np

from dualguard.utils.model import calc_shifted_loss_logits

HeadModel=Union[QwenHead,LlamaHead,GPT2Head]
TailModel=Union[QwenTail,LlamaTail,GPT2Tail]
ServerModel=Union[LlamaServer,QwenServer,GPT2Server]

def _lamp(
        lamp_step:int,
        beta:float,
        forzen_pretrained_tail_model:TailModel, 
        pt_llm:AutoModelForCausalLM, 
        dummy_x:torch.Tensor, 
        true_grads:torch.Tensor, 
        dummy_lables_logits:torch.Tensor, 
        **kwargs):
    """
    Copied from the original implementation of LAMP token swaping.
    https://github.com/eth-sri/lamp/blob/main/attack.py
    """
    dummy_x.requires_grad = True
    best_gt, best_loss = None, None
    init_loss = None
    changed = None
    batch_size, seq_len, vocab_size = dummy_lables_logits.shape
    # with tqdm(total=lamp_steps,desc="LAMP optimization") as pbar:
    for sample_idx in range(lamp_step):
        new_dummy_lables_logits = dummy_lables_logits.clone().detach()
        with torch.no_grad():
            for sen_id in range(batch_size):
                if sample_idx != 0:
                    perm_ids = np.arange(seq_len)
                    if sample_idx != 0:
                        if sample_idx % 4 == 0:  # swap two tokens
                            i, j = 1 + np.random.randint(seq_len - 2), 1 + np.random.randint(seq_len - 2)
                            perm_ids[i], perm_ids[j] = perm_ids[j], perm_ids[i]
                        elif sample_idx % 4 == 1:  # move a token to another place
                            i = 1 + np.random.randint(seq_len - 2)
                            j = 1 + np.random.randint(seq_len - 1)
                            if i < j:
                                perm_ids = np.concatenate(
                                    [perm_ids[:i], perm_ids[i + 1:j], perm_ids[i:i + 1], perm_ids[j:]])
                            else:
                                perm_ids = np.concatenate(
                                    [perm_ids[:j], perm_ids[i:i + 1], perm_ids[j:i], perm_ids[i + 1:]])
                        elif sample_idx % 4 == 2:  # move a sequence to another place
                            b = 1 + np.random.randint(seq_len - 1)
                            e = 1 + np.random.randint(seq_len - 1)
                            if b > e:
                                b, e = e, b
                            p = 1 + np.random.randint(seq_len - 1 - (e - b))
                            if p >= b:
                                p += e - b
                            if p < b:
                                perm_ids = np.concatenate(
                                    [perm_ids[:p], perm_ids[b:e], perm_ids[p:b], perm_ids[e:]])
                            elif p >= e:
                                perm_ids = np.concatenate(
                                    [perm_ids[:b], perm_ids[e:p], perm_ids[b:e], perm_ids[p:]])
                            else:
                                assert False
                        elif sample_idx % 4 == 3:  # take some prefix and put it at the end
                            i = 1 + np.random.randint(seq_len - 2)
                            perm_ids = np.concatenate(
                                [perm_ids[:1], perm_ids[i:-1], perm_ids[1:i], perm_ids[-1:]])
                    new_dummy_lables_logits[sen_id] = dummy_lables_logits[sen_id, perm_ids, :]
        new_dummy_lables_logits.requires_grad = True
        _,grad_diff_loss=_calcuate_loss(
            beta=beta,
            forzen_llm=forzen_pretrained_tail_model,
            pt_llm=pt_llm,
            dummy_x=dummy_x,
            true_grads=true_grads,
            dummy_lables_logits=new_dummy_lables_logits,
            **kwargs
            )
        if sample_idx == 0:
            init_loss = grad_diff_loss
        # pbar.set_description({'loss': grad_diff_loss, 'best_loss': best_loss, 'init_loss': init_loss})
        if (best_loss is None) or (grad_diff_loss < best_loss):
            best_gt = new_dummy_lables_logits
            best_loss = grad_diff_loss
            if sample_idx != 0:
                changed = sample_idx % 4
            # pbar.update(1)
    if not (changed is None):
        change = ['Swapped tokens', 'Moved token', 'Moved sequence', 'Put prefix at the end'][changed]
        # print(change)
    return best_gt

def _calcuate_loss(        
        beta:float,
        forzen_llm:TailModel, 
        pt_llm:AutoModelForCausalLM,
        dummy_x:torch.Tensor, 
        true_grads:torch.Tensor, 
        dummy_lables_logits:torch.Tensor, 
        with_ppl:bool=False,
        **kwargs
    ):
    if isinstance(forzen_llm,GPT2Tail):
        dummy_pred=forzen_llm(
            hidden_status_from_server=dummy_x,
            presents=kwargs.get('presents',None),
            past_key_values=kwargs.get('past_key_values',None),
            attention_mask=kwargs.get('attention_mask',None),
            head_mask=kwargs.get('head_mask',None),
            encoder_hidden_states=kwargs.get('encoder_hidden_states',None),
            encoder_attention_mask=kwargs.get('encoder_attention_mask',None),
            use_cache=kwargs.get('use_cache',False),
            output_attentions=kwargs.get('output_attentions',False),
            output_hidden_states=kwargs.get('output_hidden_states',False),
            all_self_attentions=kwargs.get('all_self_attentions',None),
            all_hidden_states=kwargs.get('all_hidden_states',None),
            all_cross_attentions=kwargs.get('all_cross_attentions',None),
            labels=None,
        )
    elif isinstance(forzen_llm,(LlamaTail,QwenTail)):
        dummy_pred=forzen_llm(
            hidden_status_from_server=dummy_x,
            attention_mask=kwargs.get('attention_mask',None),
            position_ids=kwargs.get('position_ids',None),
            past_key_values=kwargs.get('past_key_values',None),
            output_attentions=kwargs.get('output_attentions',False),
            use_cache=kwargs.get('use_cache',False),
            cache_position=kwargs.get('cache_position',None),
            all_hidden_states=kwargs.get('all_hidden_states',None),
            all_self_attns=kwargs.get('all_self_attns',None),
            return_legacy_cache=kwargs.get('return_legacy_cache',False),
            labels=None
        )
    dummy_pred:CausalLMOutputWithPast
    logits=dummy_pred.logits
    loss=calc_shifted_loss_logits(logits,torch.softmax(dummy_lables_logits,dim=-1))
    #计算loss对dummy_lables的梯度
    dummy_x_grad = torch.autograd.grad(loss,dummy_x,create_graph=True)
    # TAG gradient-matching loss
    grad_diff=0.0
    for gx, gy in zip(dummy_x_grad[0], true_grads.to(loss.device)):
        grad_diff += beta * ((gx - gy) ** 2).sum() + (1 - beta) * (torch.abs(gx - gy)).sum()
    if with_ppl:
        with torch.no_grad():
            input_ids = dummy_lables_logits.argmax(-1)
            ppl = pt_llm(input_ids=input_ids,attention_mask=kwargs.get('attention_mask',None),labels=input_ids).loss
            # print(f'ppl: {ppl}')
            grad_diff += 0.2 * ppl
    return loss,grad_diff

'''
dlg所需要的参数:
1. forzen_llm: 预训练模型
2. dummy_x: 激活量: b s v
3. dummy_lables: shifted tokens  :b s v
4. true_grads: 激活量的梯度: b s v
5. optimizer: 优化器 __class__
6. tokenizer: tokenizer
7. device: 设备
8. iters: 迭代次数
9. beta: 平滑系数

    前向传播1
    前向传播2-> dummy x
    前向传播3（tail） ->real y loss0
    前向传播3.1(pre-trained tail) -> dummy y :loss1
    
    后向传播3（tail）->梯度0(server 输出激活量的梯度)
    后向传播3.1(pre-trained tail) -> 梯度1（server 输出激活量的梯度）
    
    用梯度1和梯度0做距离损失
    
    距离损失更新dummy y
'''
def lamp_attack(
    forzen_llm:TailModel,
    pt_llm:AutoModelForCausalLM,
    dummy_x:Optional[torch.Tensor], 
    dummy_lables_logits:Optional[torch.Tensor],
    true_grads:Optional[torch.Tensor],
    optimizer:torch.optim.Optimizer.__class__=torch.optim.AdamW, 
    attack_iters:int=1000,
    lamp_freq:int=10,
    lamp_step:int=100,
    beta:float=0.85,
    **kwargs
    )->Tuple[torch.Tensor, torch.Tensor]:
    # beta=args.beta #平滑系数
    true_grads.requires_grad=False
    dummy_x.requires_grad=True
    dummy_lables_logits.requires_grad=True
    optim = optimizer([dummy_lables_logits], lr=0.3, betas=(0.9, 0.999), eps=1e-6)
    with tqdm(total=attack_iters,desc='LAMP Attack') as pbar:
        for i in range(attack_iters):
            optim.zero_grad()
            loss,grad_diff=_calcuate_loss(beta,forzen_llm,pt_llm,dummy_x,true_grads,dummy_lables_logits,with_ppl=False,**kwargs)#loss仅用于打印
            grad_diff.backward()
            optim.step()
            if i % lamp_freq == 0:
                new_dummy_lables_logits=_lamp(lamp_step,beta,forzen_llm,pt_llm,dummy_x,true_grads.clone().detach(),dummy_lables_logits,with_ppl=False,**kwargs)
                with torch.no_grad():
                # copy new_gt's data to gt
                    dummy_lables_logits.data = new_dummy_lables_logits.data
            pbar.set_description(f'iter {i+1}, loss {loss.item():.5f}, grad_diff {grad_diff.item():.5f}')
            pbar.update(1)  
            torch.cuda.empty_cache()
    return dummy_lables_logits#返回恢复后的标签 

def attack(    
    model_head:HeadModel,
    model_server:ServerModel,
    model_tail:TailModel,
    forzen_pretrained_tail_model:TailModel,
    dummy_labels_logits:torch.Tensor,
    batch:Dict[str,torch.Tensor],
    args:dict,
    tokenizer:AutoTokenizer):
    model_head.eval()
    model_tail.eval()
    forzen_pretrained_tail_model.eval()
    device = args.device if args.device is not None else "cpu"
    # print("original text:",[tokenizer.decode(batch["input"][i])for i in range(1)])
    dummy_x,true_grads,tail_output_logits,attention_mask,position_ids=forward_and_get_true_grads(model_head,model_server,model_tail,batch,device)
    if dummy_labels_logits is None:
        batch_size, seq_len, vocab_size = tail_output_logits.shape
        dummy_labels_logits = torch.softmax(torch.randn((batch_size, seq_len, vocab_size)).to(dummy_x.device), dim=-1)
    dummy_lables_logits = lamp_attack(
        args=args,
        forzen_llm=forzen_pretrained_tail_model,
        dummy_x=dummy_x,dummy_lables_logits=dummy_labels_logits,
        true_grads=true_grads,
        optimizer=torch.optim.AdamW,tokenizer=tokenizer,
        attention_mask=attention_mask,position_ids=position_ids
    )
    # print(f"attacked labels: {dummy_lables_logits}")
    return dummy_lables_logits.argmax(-1)

def attack_main(    
    model_head:HeadModel,
    model_server:ServerModel,
    model_tail:TailModel,
    forzen_pretrained_tail_model:TailModel,
    dummy_labels_logits:torch.Tensor,
    train_loader:DataLoader,
    args:dict,
    tokenizer:AutoTokenizer,
    sample_rate:float=0.2):
    model_head.eval()
    model_tail.eval()
    forzen_pretrained_tail_model.eval()
    device = args.device if args.device is not None else "cpu"
    for idx, data in enumerate(train_loader):
        if idx==int(len(train_loader)*sample_rate):
            break
        dummy_tokens=attack(
            model_head=model_head,
            model_server=model_server,
            model_tail=model_tail,
            forzen_pretrained_tail_model=forzen_pretrained_tail_model,
            dummy_labels_logits=dummy_labels_logits,
            batch=data,
            args=args,
            tokenizer=tokenizer
        )
        # 单独实验使用
        recovered_texts=[tokenizer.decode(dummy_label) for dummy_label in dummy_tokens]
        print(f"recovered text: {recovered_texts}")
        break

    
        
