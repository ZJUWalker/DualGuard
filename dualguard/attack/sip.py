'''
sip attack is based on the codebase repo: "https://github.com/StupidTrees/SplitLLM"
'''

from logging import Logger
from typing import Dict
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dualguard.usl.split_model import SplitModel
from dualguard.utils.exp import get_dataset
from dualguard.utils.model import *
from dualguard.utils.configs import SIPAttackArgs, EnvArgs

from transformers import AutoTokenizer


class GRUDRInverter(nn.Module):

    def __init__(self, n_embed, vocab_size, hidden_size=256, bidirectional=True, *args, **kwargs):
        super().__init__()
        self.gru = nn.GRU(input_size=n_embed, hidden_size=hidden_size, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            hidden_size *= 2
        self.mlp = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor):
        if x.dtype == torch.float16:
            x = x.float()
        hidden, _ = self.gru(x)  # hidden [batch_size, seq_len, n_embed]
        hidden = torch.dropout(hidden, p=0.1, train=self.training)
        return self.mlp(hidden)

    def search(self, x, base_model, beam_size=6):
        logits = self.forward(x.to(self.device))
        batch_size, seq_len, vocab_size = logits.shape
        beams = [(None, [0] * batch_size)] * beam_size
        for step in range(seq_len):
            candidates = []
            for sentence_batch, sent_score_batch in beams:
                last_token_logits = logits[:, step, :]
                topk_probs, topk_indices = torch.topk(last_token_logits, beam_size)
                topk_probs = torch.softmax(topk_probs, dim=-1)
                for k in range(beam_size):
                    prob, token = topk_probs[:, k].unsqueeze(-1), topk_indices[:, k].unsqueeze(-1)  # (batch_size, 1)
                    sents = torch.cat([sentence_batch, token], dim=1) if sentence_batch is not None else token  # (batch_size, seq++)
                    candidate_score = self._sentence_score_tokens(sents, base_model).unsqueeze(-1)  # (batch_size, 1)
                    score = prob * 5 - candidate_score
                    # print(prob.shape, candidate_score.shape, score.shape)
                    candidates.append((sents, score))
            new_list = []
            for batch in range(batch_size):
                # print(candidates)
                candidates_batch = [(c[batch, :].unsqueeze(0), score[batch, :].unsqueeze(0)) for c, score in candidates]
                # print(candidates_batch)
                candidates_batch = sorted(candidates_batch, key=lambda x: x[-1], reverse=True)
                if len(new_list) == 0:
                    new_list = candidates_batch
                else:
                    nl = []
                    for (sent, score), (sent2, score2) in zip(new_list, candidates_batch):
                        nl.append((torch.concat([sent, sent2], dim=0), torch.concat([score, score2], dim=0)))
                    new_list = nl
            beams = new_list[:beam_size]
        return beams[0][0]

    def _sentence_score_tokens(sent, model: nn.Module):
        model.eval()
        padded = sent.to(model.device).long()
        stride = 16
        scoress = []
        for i in range(int(np.ceil(len(padded) / stride))):
            outputs = model(padded[i * stride : min((i + 1) * stride, len(padded))])
            lsm = -outputs[0].log_softmax(2)
            preds = torch.zeros_like(lsm)
            preds[:, 1:] = lsm[:, :-1]
            wordscores = preds.gather(2, padded[i * stride : min((i + 1) * stride, len(padded))].unsqueeze(2)).squeeze(2).detach()
            scores = wordscores.sum(1) / wordscores.shape[1]
            scoress.append(scores)
        score = torch.cat(scoress)
        model.train()
        return score


def load_sip_model(model_path: str, model_clz: nn.Module.__class__ = None, deivce_mapping: Dict[str, str] = 'cpu', **kwargs) -> nn.Module:
    if model_clz is None:
        model_clz = GRUDRInverter
    model_dict = torch.load(model_path, map_location=deivce_mapping, weights_only=True)
    try:
        sip_model = model_clz(**kwargs)
        sip_model.load_state_dict(model_dict)
    except Exception as e:
        print(f"exception {e} load model {model_path} failed, try to load it without weights_only")
        return model_dict
    return sip_model


def sip_model_train(
    env_args: EnvArgs,
    sip_args: SIPAttackArgs,
    head_model: SplitModel,
    tokenizer: AutoTokenizer,
    attack_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    logger: Logger = None,
) -> nn.Module:
    device = env_args.device
    attack_model.cuda(device)
    head_model.cuda(device)
    # log_strs = []
    if dataloader is None:
        aux_dataset = get_dataset('wikitext', tokenizer=tokenizer, client_ids=[])
        dataloader = aux_dataset.get_dataloader_unsliced(sip_args.batch_size, 'validation')
    with tqdm(total=sip_args.epochs * len(dataloader)) as pbar:
        for epc in range(sip_args.epochs):
            head_model.train(True)
            rouge_l_f = 0
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()
                input_ids = batch['input'].to(device)
                intermediate = head_model(input_ids=input_ids)[0]
                logits = attack_model(intermediate)
                loss = calc_unshift_loss(logits, input_ids)
                loss.backward()
                optimizer.step()
                res, _, _ = evaluate_attacker_rouge(tokenizer, logits, batch)
                rouge_l_f += res['rouge-l']['f']
                pbar.set_description(f'Epoch {epc+1} | Loss {loss.item():.5f} | Rouge_Lf1 {rouge_l_f / (step + 1):.4f}')
                pbar.update(1)
            logger.info(f"Epoch {epc+1} | Loss {loss.item():.5f} | Rouge_Lf1 {rouge_l_f / len(dataloader):.4f}")

    return attack_model


def sip_model_evaluate(
    env_args: EnvArgs,
    attack_model: nn.Module,
    head_model: SplitModel,
    tokenizer: AutoTokenizer,
    valid_loader: DataLoader = None,
    logger: Logger = None,
):
    device = env_args.device
    attack_model.to(device)
    head_model.to(device)
    attack_model.eval()
    if valid_loader is None:
        aux_dataset = get_dataset('wikitext', tokenizer=tokenizer, client_ids=[])
        dataloader = aux_dataset.get_dataloader_unsliced(SIP_train_args.batch_size, 'validation')
    else:
        dataloader = valid_loader
    rouge_l_f = 0
    meteors = 0
    total_steps = len(dataloader)
    _loss = 0
    with tqdm(total=total_steps, desc="Training Progress") as pbar:
        for step, batch in enumerate(dataloader):
            _msk = batch["attention_mask"].to(device)
            input_ids = batch['input'].to(device)
            intermediate = head_model(input_ids=input_ids, attention_mask=_msk)[0]  # 得到输入和中间层结果
            logits = attack_model(intermediate)
            loss = calc_unshift_loss(logits, input_ids)
            _loss += loss.item()
            res, meteor, _ = evaluate_attacker_rouge(tokenizer, logits, batch)
            rouge_l_f += res['rouge-l']['f']
            meteors += meteor
            pbar.set_description(f'Loss: {_loss / (step + 1):.5f} | Rouge_Lf1: {rouge_l_f / (step + 1):.4f} | Meteor: {meteors / (step + 1):.4f}')
            pbar.update(1)
    return rouge_l_f / total_steps, meteors / total_steps, _loss / total_steps
