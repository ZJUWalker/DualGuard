import os
import random
from typing import Tuple
import numpy as np
import pynvml
import torch
from rouge import Rouge
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoModelForCausalLM,AutoTokenizer,PreTrainedModel,PretrainedConfig
from trl import DPOTrainer
import nltk
from nltk.translate import meteor_score
from dualguard.utils import env

#用于计算meteor指标
nltk.data.path.append(env.nltk_path)
MODEL_PATH=env.model_path

def new_mask(input_tensor: torch.Tensor) -> torch.Tensor:
    # 复制张量以保存结果
    result = input_tensor.clone()

    # 找到每一行第一个1的位置
    first_one_indices = (input_tensor == 1).float().argmax(dim=1)

    # 遍历每一行，修改结果张量
    for i, idx in enumerate(first_one_indices):
        if (input_tensor[i] == 1).any():  # 如果存在 1
            result[i, :idx] = 1

    return result


def get_embed_size(target_config: PretrainedConfig):
    n_embed = 0
    if hasattr(target_config, 'n_embd'):
        n_embed = target_config.n_embd
    elif hasattr(target_config, 'hidden_size'):
        n_embed = target_config.hidden_size
    elif hasattr(target_config, 'd_model'):
        n_embed = target_config.d_model
    return n_embed


def decode_with_extra_space(tok, sent, mask):
    strs = []
    for idx, tid in enumerate(sent):
        if mask[idx] == 0:
            return ' '.join(strs)
        strs.append(tok.decode(tid, skip_special_tokens=True))
    # print(' '.join(strs)[:20], tok.decode(sent, skip_special_tokens=True)[:20])
    return ' '.join(strs)


def decode_with_extra_space_evaluate(tok, sent, mask):
    strs = []
    for idx, tid in enumerate(sent):
        if mask[idx]:
            strs.append(tok.decode(tid, skip_special_tokens=True))
    # print(' '.join(strs)[:20], tok.decode(sent, skip_special_tokens=True)[:20])
    return ' '.join(strs)


def evaluate_attacker_rouge(tok, attacker_logits, batch,**kwargs):
    device=kwargs.get('device','cuda')
    _msk = batch['attention_mask'].to(device)
    mask = new_mask(_msk).bool().to(device)
    _input = batch['input'].to(device)
    # _input = _input[mask.any(dim=1)].cuda()  # 提取掩码中有至少一个值为 1 的行
    # attacker_logits = [t[m] for t, m in zip(attacker_logits, mask)]
    # print(len(attacker_logits))
    # print("attacker_logits.argmax(dim=-1).shape:", attacker_logits.argmax(dim=-1).shape)
    if 'input_santi_mask' in batch:  # 只评估敏感词
        mask = batch['input_santi_mask']
        zero_indexes = torch.where(mask == 0)
        masked_atk_ids = attacker_logits.argmax(dim=-1)
        masked_atk_ids[zero_indexes] = tok.unk_token_id
        masked_gt_ids = batch['input_ids'].clone()
        masked_gt_ids[zero_indexes] = tok.unk_token_id
        atk_txts = [decode_with_extra_space(tok, s) for s in masked_atk_ids]
        gt_txts = [decode_with_extra_space(tok, s) for s in masked_gt_ids]
    else:
        atk_txts = [decode_with_extra_space(tok, s, m) for s, m in zip(attacker_logits.argmax(dim=-1), mask)]
        gt_txts = [decode_with_extra_space(tok, s, m) for s, m in zip(_input,mask)]  # batch['input_text']
    rouge = calculate_rouge_text(atk_txts, gt_txts, print_comparison=False)
    meteor = calculate_meteor(atk_txts, gt_txts)
    token_acc = calculate_token_acc(tok, atk_txts, gt_txts)
    return rouge, meteor, token_acc


def evaluate_lm_rouge(tok, lm_logits, batch,**kwargs):
    device=kwargs.get('device','cuda:0')
    _mask = batch['_mask'].to(device)
    _input = batch['input'].to(device)

    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = _input[..., 1:].contiguous()
    shift_mask = _mask[..., 1:].contiguous()
    print("shift_logits.shape:", shift_logits.shape)
    print("shift_labels.shape:", shift_labels.shape)
    print("shift_mask.shape:", shift_mask.shape)

    atk_txts = [decode_with_extra_space(tok, s, m) for s, m in zip(shift_logits.argmax(dim=-1), shift_mask)]
    gt_txts = [decode_with_extra_space(tok, s, m) for s, m in zip(shift_labels, shift_mask)]  # batch['input_text']
    rouge = calculate_rouge_text(atk_txts, gt_txts, print_comparison=False)
    meteor = calculate_meteor(atk_txts, gt_txts)
    # token_acc = calculate_token_acc(tok, atk_txts, gt_txts)
    return rouge, meteor


def calc_loss(head, server, tail, valid_loader, tokenizer):
    rouge_l_f = 0
    with torch.no_grad():
        for idx, batch in enumerate(valid_loader):
            if idx % 50 == 0:
                print(f"valid batch {idx+1}/{len(valid_loader)}")
            _input = batch["input"].to('cuda')
            _target = batch["target"].to('cuda')

            hidden_states0, head_presents, _ = head(_input)

            hidden_states1, server_presents, _ = server(_input.shape, hidden_states0, head_presents)

            _logits, _loss, _ = tail(
                _input.shape,
                hidden_states1,
                server_presents,
            )
            # _loss = calc_unshift_loss(_logits, _target, _msk, args.label_smooth)
            shift_logits = _logits[..., :-1, :].contiguous()  # 去掉最后一个时间步
            shift_labels = _target[..., 1:].contiguous()     # 去掉第一个时间步
            text0 = [decode_with_extra_space(tokenizer, s) for s in shift_logits.argmax(dim=-1)]
            original_text = [decode_with_extra_space(tokenizer, s) for s in shift_labels]
            rouge = calculate_rouge_text(text0, original_text, print_comparison=False)
            rouge_l_f += rouge["rouge-l"]["f"]

    return rouge_l_f / len(valid_loader)
 

def calculate_token_acc(tok, texts, labels):
    f1_avg = 0
    for g, r in zip(texts, labels):
        generated_tokens = set(tok.tokenize(g))
        reference_tokens = set(tok.tokenize(r))
        # calculate overlap number
        overlap = len(generated_tokens & reference_tokens)
        # calculate precision and recall
        if len(generated_tokens) == 0:
            precision = 0
        else:
            precision = overlap / len(generated_tokens)
        if len(reference_tokens) == 0:
            recall = 0
        else:
            recall = overlap / len(reference_tokens)
        # calculate f1
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1_avg += f1
    if len(texts) == 0:
        return 0
    return f1_avg / len(texts)


def evaluate_attacker_mse(attacker_outputs, batch):
    return torch.nn.functional.mse_loss(attacker_outputs, batch)


def calculate_rouge(tok, logits, labels, print_comparison=False, is_tokens=False):
    if not is_tokens:
        output_texts = [tok.decode(logits.argmax(dim=-1)[i], skip_special_tokens=True) for i in
                        range(len(logits))]
    else:
        output_texts = [tok.decode(s, skip_special_tokens=True) for s in logits]
    return calculate_rouge_text(output_texts, labels, print_comparison=print_comparison)





def calculate_meteor(texts, labels):
    meteor_sum = 0
    meteor_num = 0
    for text, label in zip(texts, labels):
        meteor = meteor_score.meteor_score([label.split()], text.split())
        meteor_sum += meteor
        meteor_num += 1
    return meteor_sum / meteor_num


def calculate_rouge_text(texts, labels, print_comparison=False):
    my_rouge = Rouge()
    hyps_and_refs = zip(texts, labels)
    hyps, refs = zip(*hyps_and_refs)
    if print_comparison:
        for h, r in zip(hyps, refs):
            print(f'{r}==>{h}')
    try:
        hyps = list('<EMP>' if len(h) == 0 else h for h in hyps)
        result = my_rouge.get_scores(hyps, refs, avg=True, ignore_empty=True)  # 取一个 batch 的平均
    except Exception as e:
        print(f'rouge error {e}')
        print(hyps, refs)
        result = {'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                  'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                  'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}}
    return result


def generate(text, tokenizer, md, **kwargs):
    md.train(False)
    t = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    res = md.generate(t['input_ids'].to(md.device), attention_mask=t['attention_mask'].to(md.device), **kwargs)
    # max_length=128)#, num_beams=8, no_repeat_ngram_size=4, early_stopping=True,
    # num_return_sequences=1)
    return tokenizer.decode(res[0], skip_special_tokens=True)


def get_output(text, tokenizer, md):
    t = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    if md.type == 'encoder-decoder':
        res = md(t['input_ids'].to(md.device), attention_mask=t['attention_mask'].to(md.device),
                 decoder_input_ids=t['input_ids'].to(md.device))
    else:
        res = md(t['input_ids'].to(md.device), attention_mask=t['attention_mask'].to(md.device))

    r = tokenizer.decode(res.logits.argmax(dim=-1)[-1], skip_special_tokens=True)
    return r


def sentence_score(sentence, model, tokenizer):
    sentence = torch.LongTensor(tokenizer.encode(sentence))
    before = torch.LongTensor(
        tokenizer.encode("<|endoftext|>")
    )  # ],return_tensors='pt').input_ids[0]
    after = torch.LongTensor(
        tokenizer.encode("<|endoftext|>")
    )  # ],return_tensors='pt').input_ids# [0]

    sent = sentence

    padded = torch.cat([before, sent, after], 0).unsqueeze(0).to(model.device)
    stride = 16
    padded = padded.long()
    wordscoress, scoress, predss = [], [], []
    for i in range(int(np.ceil(len(padded) / stride))):
        outputs = model(padded[i * stride: min((i + 1) * stride, len(padded))])
        lsm = -outputs[0].log_softmax(2)
        preds = torch.zeros_like(lsm)
        preds[:, 1:] = lsm[:, :-1]
        wordscores = (
            preds.gather(
                2, padded[i * stride: min((i + 1) * stride, len(padded))].unsqueeze(2)
            )
            .squeeze(2)
            .cpu()
            .detach()
        )
        scores = wordscores.sum(1) / wordscores.shape[1]
        wordscoress.append(wordscores)
        scoress.append(scores)
        predss.append(preds.cpu().detach())

    # wordscores = torch.cat(wordscoress)
    score = torch.cat(scoress).min()
    # preds = torch.cat(predss)
    return score


def sentence_score_tokens(sent, model):
    model.train(False)
    padded = sent.to(model.device).long()
    stride = 16
    scoress = []
    for i in range(int(np.ceil(len(padded) / stride))):
        outputs = model(padded[i * stride: min((i + 1) * stride, len(padded))])
        lsm = -outputs[0].log_softmax(2)
        preds = torch.zeros_like(lsm)
        preds[:, 1:] = lsm[:, :-1]
        wordscores = (
            preds.gather(
                2, padded[i * stride: min((i + 1) * stride, len(padded))].unsqueeze(2)
            )
            .squeeze(2)
            .detach()
        )
        scores = wordscores.sum(1) / wordscores.shape[1]
        scoress.append(scores)
    # wordscores = torch.cat(wordscoress)
    score = torch.cat(scoress)
    model.train(True)
    DPOTrainer
    return score


def get_best_gpu():
    """Return gpu (:class:`torch.device`) with largest free memory."""
    assert torch.cuda.is_available()
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()

    if "CUDA_VISIBLE_DEVICES" in os.environ.keys() is not None:
        cuda_devices = [
            int(device) for device in os.environ["CUDA_VISIBLE_DEVICES"].split(',')
        ]
    else:
        cuda_devices = range(deviceCount)

    assert max(cuda_devices) < deviceCount
    deviceMemory = []
    for i in cuda_devices:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        deviceMemory.append(mem_info.free)
    deviceMemory = np.array(deviceMemory, dtype=np.int64)
    best_device_index = np.argmax(deviceMemory)
    return torch.device("cuda:%d" % (best_device_index))


def set_random_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


def calc_unshift_loss(lm_logits, labels):
    labels = labels.to(lm_logits.device)
    # do not shift
    loss_fct = CrossEntropyLoss()
    return loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))


def calc_unshift_loss_logits(lm_logits, labels):
    labels = labels.to(lm_logits.device)
    # do not shift
    loss_fct = CrossEntropyLoss()
    return loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1, labels.size(-1)))


def calc_shifted_loss_logits(lm_logits:torch.Tensor, label_logits:torch.Tensor):
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = label_logits[..., 1:, :].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # return loss_fct(shift_logits, shift_labels)
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1, shift_labels.size(-1)))


def calc_chatglm_loss(lm_logits, label_logits):
    lm_logits = lm_logits.to(torch.float32)

    # Shift so that tokens < n predict n
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = label_logits[..., 1:, :].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1, shift_labels.size(-1)))


def calc_shifted_loss(lm_logits, labels):
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def calc_circular_loss(lm_logits, labels):
    shift_logits = lm_logits[..., :, :].contiguous()
    shift_labels = torch.concat([labels[..., 1:], labels[..., 0:1]], dim=-1).contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def batched_perplexity(model, dataloader, tokenizer, stride=512):
    device = model.device
    max_len = model.config.n_positions
    # encodings = tokenizer("\n\n".join(data["text"]), return_tensors="pt")
    # text_len = encodings.input_ids.size(1)
    lls = []
    ppl_total = 0
    batch_num = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        batch_size, text_len = input_ids.shape[:2]
        for i in tqdm(range(0, text_len, batch_size * stride)):
            begin_locs, end_locs, trg_lens = [], [], []
            for j in range(batch_size):
                j = i + j * stride
                if j >= text_len:
                    break
                begin_loc = max(j + stride - max_len, 0)
                end_loc = min(j + stride, text_len)
                trg_len = end_loc - j  # may be different from stride on last loop

                begin_locs.append(begin_loc)
                end_locs.append(end_loc)
                trg_lens.append(trg_len)

            input_ids = [input_ids[:, b:e] for b, e in zip(begin_locs, end_locs)]
            target_end_locs = [sen.size(-1) for sen in input_ids]
            input_ids = [
                torch.nn.functional.pad(sen, (0, max_len - sen.size(-1)), "constant", 0) for sen in input_ids
            ]  # we dont need attention mask as long as these padded token is not involved in loss calculation
            input_ids = torch.stack(input_ids, dim=1).squeeze(0).to(device)

            target_ids = torch.ones_like(
                input_ids) * -100  # -100 is the default ingore_index value in torch.nn.CrossEntropyLoss
            for i, (b, e) in enumerate(zip(trg_lens, target_end_locs)):
                labels = input_ids[i, -b:e].clone()
                target_ids[i, -b:e] = labels

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                log_likelihood = outputs["loss"] * sum(trg_lens)

            lls.append(log_likelihood)
        ppl_total += torch.exp(sum(torch.stack(lls) / end_locs[-1]))
        batch_num += 1
    return ppl_total / batch_num


def evaluate_perplexity(model, loader, stride=1024):
    if loader is None:
        return 0
    model.train(False)
    if hasattr(model.config, 'n_positions'):
        max_length = model.config.n_positions
    elif hasattr(model.config, 'max_length'):
        max_length = model.config.max_length
    ppl = 0
    len = 0
    for batch in tqdm(loader, desc='evaluating perplexity'):
        input_ids = batch['input_ids'].to(model.device)
        seq_len = input_ids.size(1)
        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = input_ids[:, begin_loc:end_loc].to(model.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss
            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl += torch.exp(torch.stack(nlls).mean())
        len += 1
    model.train(True)
    return ppl / len


def evaluate_accuracy(model, loader):
    if loader is None:
        return 0
    model.train(False)
    acc = 0
    itm = 0
    for batch in loader:
        if 'input_ids' not in batch:
            input_ids = batch['input'].to(model.device)
        else:
            input_ids = batch['input_ids'].to(model.device)
        labels = batch['labels'].to(model.device)  # (batch_size, )

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits.view(-1, model.num_labels)  # (, num_labels)
            preds = torch.argmax(logits, dim=-1)
            labels = labels.view(-1)
            acc += (preds == labels).sum() / labels.numel()
            itm += 1
    model.train(True)
    return acc / itm

def dist_corr(x, y):
    """
    Compute the distance correlation function
    """
    x = x.float()
    y = y.float()
    n = x.size(0)
    a = torch.cdist(x, x)
    b = torch.cdist(y, y)
    A = a - a.mean(dim=0) - a.mean(dim=1)[:, None] + a.mean()
    B = b - b.mean(dim=0) - b.mean(dim=1)[:, None] + b.mean()
    dcov2 = (A * B).sum() / n ** 2
    dvar = (A ** 2).sum() / n ** 2
    dvar2 = (B ** 2).sum() / n ** 2
    dcor = dcov2 / (torch.sqrt(dvar) * torch.sqrt(dvar2))
    return dcor

def get_embedding_layer(llm):
    if hasattr(llm, 'model'):
        if hasattr(llm.model, 'embed_tokens'):
            return llm.model.embed_tokens
    elif hasattr(llm, 'transformer'):
        if hasattr(llm.transformer, 'wte'):
            return llm.transformer.wte
        elif hasattr(llm.transformer, 'embedding'):
            return llm.transformer.embedding
        elif hasattr(llm.transformer, 'word_embeddings'):
            return llm.transformer.word_embeddings
    elif hasattr(llm, 'bert'):
        return llm.bert.embeddings.word_embeddings
    elif hasattr(llm, 'embed_tokens'):
        return llm.embed_tokens
    elif hasattr(llm, 'word_embeddings'):
        return llm.word_embeddings
    elif hasattr(llm, 'embedding'):
        return llm.embedding
    return None

def get_vocab_size(config:PretrainedConfig):
    if hasattr(config, 'vocab_size'):
        return config.vocab_size

#获取所有模型的模型名
def list_models(keyword:str=None):
    model_paths=[]
    prefixs=os.listdir(MODEL_PATH)
    for prefix in prefixs:
        p=os.path.join(MODEL_PATH,prefix)
        model_names=[os.path.join(prefix,x) for x in os.listdir(p)]
        for name in model_names:
            model_paths.append(name)
    res=list(filter(lambda x: keyword.lower() in x.lower(), model_paths)) if keyword else model_paths
    # print(res)
    return res

#加载模型
def load_model_and_tokenizer(model_name:str='gpt2')->Tuple[PreTrainedModel,AutoTokenizer]:
    model_path=os.path.join(MODEL_PATH,model_name)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path,attn_implementation="eager")#必须用eager模式
    return model,tokenizer

#加载模型
def load_model(model_name:str='gpt2')->PreTrainedModel:
    model_path=os.path.join(MODEL_PATH,model_name)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path,attn_implementation="eager")#必须用eager模式
    return model


def load_tokenizer(model_name:str='gpt2')->AutoTokenizer:
    model_name=list_models(model_name)[0]
    model_path=os.path.join(MODEL_PATH,model_name)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
