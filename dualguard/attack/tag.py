# import logging
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from typing import Tuple, Dict, Union, Optional

from peft import PeftModelForCausalLM
from typing import Union

# from dualguard.attack.tag import forward_and_get_true_grads
from dualguard.usl import *
from dualguard.utils.model import calc_shifted_loss_logits

# from torchviz import make_dot

HeadModel=Union[QwenHead,LlamaHead,GPT2Head]
TailModel=Union[QwenTail,LlamaTail,GPT2Tail]
ServerModel=Union[LlamaServer,QwenServer,GPT2Server]

def forward_and_get_true_grads(model_head:HeadModel,model_server:ServerModel,model_tail:TailModel,data,device,pt_tail:TailModel=None):
     # The client interacts with the server in turn
    _input_ids = data["input"].to(device)
    _target = data["input"].to(device)
    attention_mask = data["attention_mask"].to(device)
    _msk = data["_mask"].to(device)
    with torch.no_grad():
        temp_outputs=model_head(input_ids=_input_ids, attention_mask=attention_mask)
    model_head.to('cpu')
    total_loss=0
    if pt_tail is not None:
        pt_tail.train()
        for n,p in pt_tail.named_parameters():
            if p.requires_grad:
                p.requires_grad=False
    if isinstance(model_server,GPT2Server) or (isinstance(model_server,PeftModelForCausalLM) and isinstance(model_server.base_model.model,GPT2Server)):
        with torch.no_grad():
            temp_outputs=model_server(
                hidden_status_from_head=temp_outputs[0],
                presents=temp_outputs[1],
                past_key_values=temp_outputs[2],
                attention_mask=temp_outputs[3],
                head_mask=temp_outputs[4],
                encoder_hidden_states=temp_outputs[5],
                encoder_attention_mask=temp_outputs[6],
                use_cache=False,
                output_attentions=temp_outputs[8],
                output_hidden_states=temp_outputs[9],
                all_self_attentions=temp_outputs[10],
                all_hidden_states=temp_outputs[11],
                all_cross_attentions=temp_outputs[12],
            )
        dummy_x:torch.Tensor = temp_outputs[0].clone().detach().requires_grad_(True)
        server_hidden_states=temp_outputs[0].clone().detach().requires_grad_(True)
        model_server.to('cpu')
        tail_output=model_tail.forward(
                hidden_status_from_server=server_hidden_states,
                presents=temp_outputs[1],
                past_key_values=None,
                attention_mask=temp_outputs[3],
                head_mask=temp_outputs[4],
                encoder_hidden_states=temp_outputs[5],
                encoder_attention_mask=temp_outputs[6],
                use_cache=False,
                output_attentions=temp_outputs[8],
                output_hidden_states=temp_outputs[9],
                all_self_attentions=temp_outputs[10],
                all_hidden_states=temp_outputs[11],
                all_cross_attentions=temp_outputs[12],
                labels=_target,
            )
        total_loss=tail_output.loss
        if pt_tail is not None:
            pt_tail_output=pt_tail.forward(
                hidden_status_from_server=server_hidden_states,
                presents=temp_outputs[1],
                past_key_values=None,
                attention_mask=temp_outputs[3],
                head_mask=temp_outputs[4],
                encoder_hidden_states=temp_outputs[5],
                encoder_attention_mask=temp_outputs[6],
                use_cache=False,
                output_attentions=temp_outputs[8],
                output_hidden_states=temp_outputs[9],
                all_self_attentions=temp_outputs[10],
                all_hidden_states=temp_outputs[11],
                all_cross_attentions=temp_outputs[12],
                labels=_target,
            )
            total_loss+=70/pt_tail_output.loss
        attention_mask=temp_outputs[3]
        position_ids=None
    else:
        with torch.no_grad():
            temp_outputs= model_server(
                hidden_status_from_head=temp_outputs[0],
                attention_mask=temp_outputs[1],#这里需要用output中的attention_mask，不能用data中的attention_mask，因为head模型会对attention_mask进行修改
                position_ids=temp_outputs[2],
                # past_key_values=None,
                output_attentions=temp_outputs[4],
                use_cache=False,
                cache_position=temp_outputs[6],
                all_hidden_states=temp_outputs[7],
                all_self_attns=temp_outputs[8],
                return_legacy_cache=temp_outputs[9],
                )
        dummy_x:torch.Tensor = temp_outputs[0].clone().detach().requires_grad_(True)
        server_hidden_states=temp_outputs[0].clone().detach().requires_grad_(True)
        model_server.to('cpu')
        tail_output= model_tail.forward(
                hidden_status_from_server=server_hidden_states,
                attention_mask=temp_outputs[1],
                position_ids=temp_outputs[2],
                # past_key_values=None,
                output_attentions=temp_outputs[4],
                use_cache=False,
                cache_position=temp_outputs[6],
                all_hidden_states=temp_outputs[7],
                all_self_attns=temp_outputs[8],
                return_legacy_cache=temp_outputs[9],
                labels=_target,
            )
        total_loss=tail_output.loss
        if pt_tail is not None:
            pt_tail_output=pt_tail.forward(
                hidden_status_from_server=server_hidden_states,
                attention_mask=temp_outputs[1],
                position_ids=temp_outputs[2],
                # past_key_values=None,
                output_attentions=temp_outputs[4],
                use_cache=False,
                cache_position=temp_outputs[6],
                all_hidden_states=temp_outputs[7],
                all_self_attns=temp_outputs[8],
                return_legacy_cache=temp_outputs[9],
                labels=_target,
            )
            total_loss+=70/pt_tail_output.loss
        attention_mask=temp_outputs[1]
        position_ids=temp_outputs[2]
    # tail_output:CausalLMOutputWithPast
    total_loss.backward()
    true_grads=server_hidden_states.grad.clone().detach()
    if pt_tail is not None: 
        pt_tail.eval()
    return dummy_x,true_grads,tail_output.logits,attention_mask,position_ids


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
def dlg_attack(forzen_llm:TailModel,
               dummy_x:Optional[torch.Tensor], 
               dummy_lables_logits:Optional[torch.Tensor],
               true_grads:Optional[torch.Tensor],
               optimizer:torch.optim.Optimizer.__class__=torch.optim.AdamW, 
               iters:int=10000, 
               beta:float=0.85, 
               **kwargs)->Tuple[torch.Tensor, torch.Tensor]:
    true_grads.requires_grad=False
    dummy_x.requires_grad=True
    dummy_lables_logits=dummy_lables_logits.detach().requires_grad_(True)
    optim=optimizer([dummy_lables_logits],lr=0.1)
    with tqdm(total=iters) as pbar:
        for i in range(iters):
            optim.zero_grad()
            if isinstance(forzen_llm, GPT2Tail) or (isinstance(forzen_llm, PeftModelForCausalLM) and isinstance(forzen_llm.base_model.model, GPT2Tail)):
                dummy_pred=forzen_llm(
                    hidden_status_from_server=dummy_x,
                    presents=kwargs.get('presents',None),
                    past_key_values=kwargs.get('past_key_values',None),
                    attention_mask=kwargs.get('attention_mask',None),
                    head_mask=kwargs.get('head_mask',None),
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    use_cache=False,
                    output_attentions=kwargs.get('output_attentions',False),
                    output_hidden_states=kwargs.get('output_hidden_states',False),
                    all_self_attentions=kwargs.get('all_self_attentions',None),
                    all_hidden_states=kwargs.get('all_hidden_states',None),
                    all_cross_attentions=kwargs.get('all_cross_attentions',None),
                    labels=None,
                )
            else:
                dummy_pred=forzen_llm(
                    hidden_status_from_server=dummy_x,
                    attention_mask=kwargs.get('attention_mask',None),
                    position_ids=kwargs.get('position_ids',None),
                    past_key_values=None,
                    output_attentions=kwargs.get('output_attentions',False),
                    use_cache=False,
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
            grad_diff.backward()
            optim.step()
            pbar.set_description(f'TAG iter {i}, loss {loss.item():.5f}, grad_diff {grad_diff.item():.5f},mem: {torch.cuda.memory_allocated(grad_diff.device) / 1024 ** 3:.2f}G')
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
    tokenizer:AutoTokenizer)->torch.Tensor:
    model_head.eval()
    model_tail.train()
    forzen_pretrained_tail_model.eval()
    device = args.device if args.device is not None else "cpu"
    iters=args.attack_iters #单次恢复迭代次数
    beta=args.beta #平滑系数
    # print("original text:",[tokenizer.decode(batch["input"][i])for i in range(1)])
    dummy_x,true_grads,tail_output_logits,attention_mask,position_ids=forward_and_get_true_grads(model_head,model_server,model_tail,batch,device)
    if dummy_labels_logits is None:
        batch_size, seq_len, vocab_size = tail_output_logits.shape
        dummy_labels_logits = torch.softmax(torch.randn((batch_size, seq_len, vocab_size)).to(dummy_x.device), dim=-1)
    dummy_lables_logits = dlg_attack(forzen_llm=forzen_pretrained_tail_model,
                dummy_x=dummy_x,dummy_lables_logits=dummy_labels_logits,
                true_grads=true_grads,
                optimizer=torch.optim.AdamW,tokenizer=tokenizer,
                device=device,iters=iters,beta=beta,attention_mask=attention_mask,position_ids=position_ids)
    return dummy_lables_logits.argmax(-1) #b x s

        
