import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer,PreTrainedModel

from dualguard.attack import lamp, tag,bisr
from dualguard.usl.llama.llama_split import *
from dualguard.usl.qwen.qwen_split import *
from dualguard.usl.gpt.gpt2_split import *
from dualguard.usl.split_model import SplitModel
from dualguard.utils.configs import BISRAttackArgs, EnvArgs, LampAttackArgs, SMAAttackArgs, TAGAttackArgs, USLTrainArgs
from dualguard.utils.model import calc_unshift_loss, calculate_meteor, calculate_rouge_text, decode_with_extra_space_evaluate,evaluate_attacker_rouge

from typing import Union,Dict
from tqdm import tqdm

HeadModel=Union[QwenHead,LlamaHead,GPT2Head]
TailModel=Union[QwenTail,LlamaTail,GPT2Tail]
ServerModel=Union[QwenServer,LlamaServer,GPT2Server]

def _to(device,*args):
    for model in args:
        model.to(device)
        
def _cal_rouge_l_f_and_meteor(tokenizer:AutoTokenizer,recovered_tks:torch.Tensor, batch:Dict[str,torch.Tensor], device):
    target=batch['input_ids'].to(device)
    atten_mask=batch['attention_mask'].to(device)
    shift_recovered_tks=recovered_tks[...,1:].contiguous()
    shift_labels = target[..., 1:].contiguous()     # 去掉第一个时间步
    shift_mask = atten_mask[..., 1:].contiguous()
    atk_txts = [decode_with_extra_space_evaluate(tokenizer, s, m) for s, m in zip(shift_recovered_tks, shift_mask)]
    gt_txts = [decode_with_extra_space_evaluate(tokenizer, s, m) for s, m in zip(shift_labels, shift_mask)]
    rouge = calculate_rouge_text(atk_txts, gt_txts, print_comparison=False)
    meteor = calculate_meteor(atk_txts, gt_txts)
    return rouge, meteor

#单个batch的攻击
def _sip_attack(env_args:EnvArgs, attack_model:nn.Module, head_model:SplitModel,tokenizer:AutoTokenizer, valid_loader:DataLoader=None):
    device = env_args.device
    attack_model.to(device)
    head_model.to(device)
    attack_model.eval()
    rouge_l_f = 0  # 初始化 Rouge_Lf1
    meteors = 0  # 初始化 Meteor
    # total_steps = int(len(valid_loader)*0.5)  # 获取总步数
    total_steps=1
    total_loss = 0
    # 创建 tqdm 进度条
    with tqdm(total=total_steps, desc="Attack Progress") as pbar:
        for step, batch in enumerate(valid_loader):
            if step >= total_steps:
                break
            # input_mask = new_mask(_msk).to('cuda')
            input_ids=batch['input'].to(device)
            _mask = batch["attention_mask"].to(device)
            intermediate = head_model(input_ids=input_ids,attention_mask=_mask)[0]  # 得到输入和中间层结果
            logits = attack_model(intermediate)

            loss = calc_unshift_loss(logits, input_ids)
            total_loss += loss.item()
            # print(f'orginal text:{tokenizer.batch_decode(input_ids, skip_special_tokens=True)}')
            # print(f'attack text:{tokenizer.batch_decode(logits.argmax(dim=-1), skip_special_tokens=True)}')
            res, meteor, _ = evaluate_attacker_rouge(tokenizer, logits, batch)
            rouge_l_f += res['rouge-l']['f']
            meteors += meteor
            # 更新 tqdm 描述信息
            pbar.set_description(
                f'Loss: {total_loss / (step + 1):.5f} | Rouge_Lf1: {rouge_l_f / (step + 1):.4f} | Meteor: {meteors / (step + 1):.4f}'
            )
            pbar.update(1) 
            torch.cuda.empty_cache()
    return rouge_l_f / total_steps, meteors / total_steps, total_loss / total_steps

def _tag_attack(
    env_args:EnvArgs,
    tag_args:TAGAttackArgs,
    model_head:HeadModel,
    model_server:ServerModel,
    model_tail:TailModel,
    forzen_pretrained_tail_model:TailModel,
    dummy_labels_logits:torch.Tensor,
    batch:Dict[str,torch.Tensor],
    tokenizer:AutoTokenizer
    ):
    model_head.eval()
    model_tail.train()
    model_server.eval()
    forzen_pretrained_tail_model.eval()
    device = env_args.device
    iters=tag_args.attack_iters #单次恢复迭代次数
    beta=tag_args.beta #平滑系数
    # print("original text:",[tokenizer.decode(batch["input"][i])for i in range(1)])
    if tag_args.is_warmup_usl:
        dummy_x,true_grads,tail_output_logits,attention_mask,position_ids=tag.forward_and_get_true_grads(model_head,model_server,model_tail,batch,device,forzen_pretrained_tail_model)
    else:
        dummy_x,true_grads,tail_output_logits,attention_mask,position_ids=tag.forward_and_get_true_grads(model_head,model_server,model_tail,batch,device)
    if dummy_labels_logits is None:
        # print('tag: randomly generate dummy labels logits')
        batch_size, seq_len, vocab_size = tail_output_logits.shape
        dummy_labels_logits = torch.softmax(torch.randn((batch_size, seq_len, vocab_size)).to(dummy_x.device), dim=-1)
    dummy_lables_logits = tag.dlg_attack(
        forzen_llm=forzen_pretrained_tail_model,
        dummy_x=dummy_x,
        dummy_lables_logits=dummy_labels_logits,
        true_grads=true_grads,
        optimizer=torch.optim.Adam,
        tokenizer=tokenizer,
        device=device,
        iters=iters,
        beta=beta,
        attention_mask=attention_mask,
        position_ids=position_ids
    )
    return dummy_lables_logits.argmax(-1) #b x s

def _lamp_attack(
    env_args:EnvArgs,
    lamp_args:LampAttackArgs,
    model_head:HeadModel,
    model_server:ServerModel,
    model_tail:TailModel,
    forzen_pretrained_tail_model:TailModel,
    pt_llm:PreTrainedModel,
    dummy_labels_logits:torch.Tensor,
    batch:Dict[str,torch.Tensor],
    tokenizer:AutoTokenizer
    ):
    model_head.eval()
    model_server.eval()
    model_tail.eval()
    forzen_pretrained_tail_model.eval()
    device = env_args.device
    # print("original text:",[tokenizer.decode(batch["input"][i])for i in range(1)])
    if lamp_args.is_warmup_usl:
        dummy_x,true_grads,tail_output_logits,attention_mask,position_ids=lamp.forward_and_get_true_grads(model_head,model_server,model_tail,batch,device,forzen_pretrained_tail_model)
    else:
        dummy_x,true_grads,tail_output_logits,attention_mask,position_ids=lamp.forward_and_get_true_grads(model_head,model_server,model_tail,batch,device)
    if dummy_labels_logits is None:
        batch_size, seq_len, vocab_size = tail_output_logits.shape
        dummy_labels_logits = torch.softmax(torch.randn((batch_size, seq_len, vocab_size)).to(dummy_x.device), dim=-1)
    dummy_lables_logits = lamp.lamp_attack(
        forzen_llm=forzen_pretrained_tail_model,
        pt_llm=None,
        dummy_x=dummy_x,
        dummy_lables_logits=dummy_labels_logits,
        true_grads=true_grads,
        optimizer=torch.optim.Adam,
        attack_iters=lamp_args.attack_iters,
        lamp_freq=lamp_args.lamp_interval,
        lamp_step=lamp_args.lamp_iters,
        beta=lamp_args.beta,
        attention_mask=attention_mask,
        position_ids=position_ids,
        tokenizer=tokenizer,
    )
    # print(f"attacked labels: {dummy_lables_logits}")
    return dummy_lables_logits.argmax(-1)

def _bisr_attack(
    env_args:EnvArgs,
    bisr_args:BISRAttackArgs,
    sip_model:nn.Module,
    head_model:HeadModel,
    tail_model:TailModel,
    forzen_head_model:HeadModel,
    forzen_tail_model:TailModel,
    server_model:ServerModel,
    tokenizer:AutoTokenizer,
    batch:Dict[str,torch.Tensor],
    all_embeds:torch.Tensor,
):
    device=env_args.device
    _to(device,sip_model,head_model,tail_model,forzen_head_model,forzen_tail_model,server_model)
    recovered_tks = bisr.bisr_attack(
        sip_model=sip_model,
        head_model=head_model,
        tail_model=tail_model,
        forzen_head_model=forzen_head_model,
        forzen_tail_model=forzen_tail_model,
        server_model=server_model,
        tokenizer=tokenizer,
        batch=batch,
        all_embeds=all_embeds,
        device=device,
        dlg_iters=bisr_args.attack_iters,
        beta=bisr_args.beta,
        sma_iters=bisr_args.sma_iters,
        is_warmup_usl=bisr_args.is_warmup_usl
    )
    return recovered_tks
#----------------------------------------评估
def sip_attack_evaluate(
    env_args:EnvArgs,
    head_model:HeadModel,
    attack_model:nn.Module,
    tokenizer:AutoTokenizer,
    data_loader:DataLoader,
):
    device = env_args.device
    _to(device,head_model,attack_model)
    #初始化参数
    return _sip_attack(
        env_args=env_args,
        attack_model=attack_model,
        lm_net_Head=head_model,
        tokenizer=tokenizer,
        valid_loader=data_loader
    )
    pass

def tag_attack_evaluate(
    env_args:EnvArgs,
    tag_args:TAGAttackArgs,
    head_model:HeadModel,
    tail_model:TailModel,
    forzen_tail_model:TailModel,
    server_model:ServerModel,
    tokenizer:AutoTokenizer,
    data_loader:DataLoader,
    ):
    device = env_args.device
    _to(device,head_model,tail_model,forzen_tail_model,server_model)
    #初始化参数
    rouge_l_f=0
    meteor=0
    total_step=tag_args.sample_num
    for idx, batch in enumerate(data_loader):
        if idx >= total_step:
            break
        recovered_tks=_tag_attack(
            env_args=env_args,
            tag_args=tag_args,
            model_head=head_model,
            model_server=server_model,
            model_tail=tail_model,
            forzen_pretrained_tail_model=forzen_tail_model,
            dummy_labels_logits=None,
            batch=batch,
            tokenizer=tokenizer
        )
        _rouge, _meteor = _cal_rouge_l_f_and_meteor(tokenizer,recovered_tks, batch, device)
        rouge_l_f += _rouge["rouge-l"]["f"]
        meteor +=_meteor
    return rouge_l_f/total_step,meteor/total_step
    pass    

def lamp_attack_evaluate(    
    env_args:EnvArgs,
    lamp_args:LampAttackArgs,
    pt_llm:PreTrainedModel,
    head_model:HeadModel,
    tail_model:TailModel,
    forzen_tail_model:TailModel,
    server_model:ServerModel,
    tokenizer:AutoTokenizer,
    data_loader:DataLoader,
    ):
    device = env_args.device
    _to(device,head_model,tail_model,forzen_tail_model,server_model)
    #初始化参数
    rouge_l_f=0
    meteor=0
    total_step=lamp_args.sample_num
    for idx, batch in enumerate(data_loader):
        if idx >= total_step:
            break
        recovered_tks=_lamp_attack(
            env_args=env_args,
            lamp_args=lamp_args,
            model_head=head_model,
            model_server=server_model,
            model_tail=tail_model,
            forzen_pretrained_tail_model=forzen_tail_model,
            pt_llm=None,
            dummy_labels_logits=None,
            batch=batch,
            tokenizer=tokenizer
        )
        _rouge, _meteor = _cal_rouge_l_f_and_meteor(tokenizer,recovered_tks, batch, device)
        rouge_l_f += _rouge["rouge-l"]["f"]
        meteor +=_meteor
    return rouge_l_f/total_step,meteor/total_step
    pass    

def bisr_attack_evaluate(
    env_args:EnvArgs,
    bisr_args:BISRAttackArgs,
    sip_model:nn.Module,
    head_model:HeadModel,
    tail_model:TailModel,
    forzen_head_model:HeadModel,
    forzen_tail_model:TailModel,
    server_model:ServerModel,
    tokenizer:AutoTokenizer,
    data_loader:DataLoader,
):
    device = env_args.device
    _to(device,sip_model,head_model,tail_model,forzen_head_model,forzen_tail_model,server_model)
    embedding_layer=head_model.get_input_embeddings()
    all_words = torch.tensor(list([i for i in range(head_model.config.vocab_size)])).to(device)
    all_embeds = embedding_layer(all_words)#将所有的token的embedding取出来
    rouge_l_f=0
    meteor=0
    total_step=bisr_args.sample_num
    for idx, batch in enumerate(data_loader):
        if idx >= total_step:
            break
        recovered_tks=_bisr_attack(
            env_args=env_args,
            bisr_args=bisr_args,
            sip_model=sip_model,
            head_model=head_model,
            tail_model=tail_model,
            forzen_head_model=forzen_head_model,
            forzen_tail_model=forzen_tail_model,
            server_model=server_model,
            tokenizer=tokenizer,
            batch=batch,
            all_embeds=all_embeds,
        )
        _rouge, _meteor = _cal_rouge_l_f_and_meteor(tokenizer,recovered_tks, batch, device)
        rouge_l_f += _rouge["rouge-l"]["f"]
        meteor +=_meteor
    return rouge_l_f/total_step,meteor/total_step
    pass

def connet_to_pt_tail_evaluate(
    env_args:EnvArgs,
    head_model:HeadModel,
    server_model:ServerModel,
    pt_tail_model:TailModel,
    tokenizer:AutoTokenizer,
    data_loader:DataLoader
):
    device = env_args.device
    _to(device,head_model,server_model,pt_tail_model)
    from dualguard.experiment.train.usl_formal_train import evaluate_usl
    loss,ppl,rouge_l_f,meteor=evaluate_usl(
        usl_args=USLTrainArgs(log_interval=50),
        env_args=env_args,
        head_model=head_model,
        server_model=server_model,
        tail_model=pt_tail_model,
        valid_loader=data_loader,
        tokenizer=tokenizer,
        output_similarity=True
        )
    return loss,ppl,rouge_l_f,meteor
    