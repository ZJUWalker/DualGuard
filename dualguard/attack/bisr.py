"""
this attack is based on the codebase repo: "https://github.com/StupidTrees/SplitLLM"
"""

import torch
from tqdm import tqdm
from transformers import AutoTokenizer


from typing import Union
from dualguard.usl.llama.llama_split import *
from dualguard.usl.qwen.qwen_split import *
from dualguard.usl.llama.llama_split import LlamaHead, LlamaTail
from dualguard.usl.qwen.qwen_split import QwenHead, QwenTail
from dualguard.usl.gpt.gpt2_split import GPT2Head, GPT2Tail, GPT2Server

from dualguard.attack import tag
from typing import Union


HeadModel = Union[QwenHead, LlamaHead, GPT2Head]
TailModel = Union[QwenTail, LlamaTail, GPT2Tail]
ServerModel = Union[LlamaServer, QwenServer, GPT2Server]


# 单个样本或batch的攻击
def sma_attack(
    forzen_head_model: HeadModel,
    dummy_label_x: torch.Tensor,  # batch_size x seq_len
    real_activation_x: torch.Tensor,  # 全局不变
    optimizer: torch.optim.Optimizer.__class__,
    iters: int,
    **kwargs,
):
    if not real_activation_x.is_leaf:
        real_activation_x = real_activation_x.clone().detach().to(forzen_head_model.device)
    real_activation_x.requires_grad = True  # 设为True但是不会更新他的值
    forzen_head_model.eval()
    batch_size, seq_len = real_activation_x.shape[:-1]
    embedding_layer = forzen_head_model.get_input_embeddings()
    if dummy_label_x is None:
        dummy = torch.randint(0, forzen_head_model.config.vocab_size, size=(batch_size, seq_len), dtype=torch.long).to(forzen_head_model.device)
    else:
        dummy = dummy_label_x.clone().detach().to(forzen_head_model.device)
    dummy_embedding_x = embedding_layer(dummy)
    if dummy_embedding_x.dtype == torch.float16:
        dummy_embedding_x = dummy_embedding_x.float()
    dummy_embedding_x = dummy_embedding_x.clone().detach().requires_grad_(True)
    optim = optimizer([dummy_embedding_x], lr=1e-4, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)
    with tqdm(total=iters) as pbar:
        for i in range(iters):
            optim.zero_grad()
            dummy_activation_x = forzen_head_model(inputs_embeds=dummy_embedding_x)[0]
            if dummy_activation_x.dtype == torch.float16:
                dummy_activation_x = dummy_activation_x.float()
            loss = 0
            for x, y in zip(dummy_activation_x, real_activation_x):
                loss += 0.85 * ((x - y) ** 2).sum()  # +0.15*(torch.abs(x - y)).sum()
            loss.backward()
            optim.step()
            pbar.set_description(f'Iter: {i}/{iters}  matching Loss: {loss.item()}')
            pbar.update(1)
            torch.cuda.empty_cache()
    return dummy_embedding_x


def bisr_attack(
    sip_model: nn.Module,
    head_model: HeadModel,
    tail_model: TailModel,
    forzen_head_model: HeadModel,
    forzen_tail_model: TailModel,
    server_model: ServerModel,
    tokenizer: AutoTokenizer,
    batch: Dict[str, torch.Tensor],
    all_embeds: torch.Tensor,
    device: torch.device = 'cuda:0',
    dlg_iters: int = 500,
    beta: float = 0.85,
    sma_iters: int = 20,
    is_warmup_usl: bool = False,
):
    input_ids = batch['input_ids'].to(device)
    atten_mask = batch['attention_mask'].to(device)
    batch_size, _ = input_ids.shape
    h2s_output = head_model.forward(input_ids=input_ids, attention_mask=atten_mask)
    h2s_ac = h2s_output[0]
    inited_dummy_x = sip_model(h2s_ac)  # bsz,seq_len,v
    inited_dummy_y_logits = inited_dummy_x.detach().requires_grad_(True)
    if is_warmup_usl:
        dummy_x, true_grads, tail_output_logits, attention_mask, position_ids = tag.forward_and_get_true_grads(
            head_model, server_model, tail_model, batch, device, forzen_tail_model
        )
    else:
        dummy_x, true_grads, tail_output_logits, attention_mask, position_ids = tag.forward_and_get_true_grads(
            head_model, server_model, tail_model, batch, device
        )
    if inited_dummy_y_logits is None:
        # print('tag: randomly generate dummy labels logits')
        batch_size, seq_len, vocab_size = tail_output_logits.shape
        inited_dummy_y_logits = torch.softmax(torch.randn((batch_size, seq_len, vocab_size)).to(dummy_x.device), dim=-1)
    dummy_y_labels = tag.dlg_attack(
        forzen_llm=forzen_tail_model,
        dummy_x=dummy_x,
        dummy_lables_logits=inited_dummy_y_logits,
        true_grads=true_grads,
        optimizer=torch.optim.Adam,
        tokenizer=tokenizer,
        device=device,
        iters=dlg_iters,
        beta=beta,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    dummy_x_tokens = dummy_y_labels
    dummy_embedding_x = sma_attack(
        forzen_head_model=forzen_head_model, dummy_label_x=dummy_x_tokens, real_activation_x=h2s_ac, optimizer=torch.optim.Adam, iters=sma_iters
    )
    cosine_similarities = torch.matmul(dummy_embedding_x, all_embeds.transpose(0, 1))
    recovered_tks = torch.softmax(cosine_similarities, -1)
    recovered_tks = torch.argmax(recovered_tks, -1)
    return recovered_tks
