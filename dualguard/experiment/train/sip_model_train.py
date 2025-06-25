import os

from typing import Union
import torch

from transformers.models.gpt2 import GPT2LMHeadModel
from transformers.models.llama import LlamaForCausalLM
from transformers.models.qwen2 import Qwen2ForCausalLM

from dualguard.attack.sip import sip_model_train, sip_model_evaluate, GRUDRInverter
from dualguard.utils.configs import EnvArgs, LogArgs, SIPTrainArgs
from dualguard.utils.exp import get_dataset
from dualguard.utils.logger import create_logger
from dualguard.utils.model import load_model_and_tokenizer, get_embed_size, set_random_seed
from dualguard.usl import *
from dualguard.experiment.method_config import *


def _init_sip(env_args: EnvArgs, sip_args: SIPTrainArgs):
    set_random_seed(env_args.random_seed)
    device = env_args.device
    # 加载模型和tokenizer
    model, tokenizer = load_model_and_tokenizer(sip_args.model_name)
    split_model: Union[LlamaSplitModel, QwenSplitModel, GPT2SplitModel] = None
    # head_layer_num=sip_args.split_point
    if isinstance(model, GPT2LMHeadModel):
        split_model = GPT2SplitModel(model, split_config=SplitModelConfig(3, -1, 3, with_server=False, logicl_load=True))
    elif isinstance(model, LlamaForCausalLM):
        split_model = LlamaSplitModel(model, split_config=SplitModelConfig(2, -1, 2, with_server=False, logicl_load=True))
    elif isinstance(model, Qwen2ForCausalLM):
        split_model = QwenSplitModel(model, split_config=SplitModelConfig(3, -1, 3, with_server=False, logicl_load=True))
    split_model.train()
    split_model.to(device)
    head_model = split_model.head_model
    # load dataset
    sip_dataset = get_dataset(sip_args.aux_dataset, tokenizer=tokenizer, client_ids=[])
    sip_train_loader = sip_dataset.get_dataloader_unsliced(sip_args.batch_size, 'train')
    sip_test_loader = sip_dataset.get_dataloader_unsliced(sip_args.batch_size, 'test')
    config = head_model.config
    vocab_size = config.vocab_size
    n_embd = get_embed_size(config)
    # init attack model
    attack_model = GRUDRInverter(n_embed=n_embd, vocab_size=vocab_size, hidden_size=256, bidirectional=False)
    attack_model.train()
    attack_model.to(device)
    # print(attack_model)
    optimizer = sip_args.optimizer(attack_model.parameters(), lr=1e-3, weight_decay=1e-5)
    return head_model, attack_model, tokenizer, optimizer, sip_train_loader, sip_test_loader


# SIP model training and evaluation
if __name__ == '__main__':
    # init env and args
    env_args = EnvArgs(device='cuda:1')
    sip_args = SIPTrainArgs(epochs=2, aux_dataset='wikitext', model_name=LLAMA3, batch_size=8)
    log_args = LogArgs(log_file_name=f'sip_model_train_{sip_args.model_name.split("/")[-1]}.log')
    logger = create_logger(log_args)
    logger.info(f'SIP model training and evaluation with args: {sip_args}')
    head_model, attack_model, tokenizer, optimizer, sip_train_loader, sip_test_loader = _init_sip(env_args, sip_args)
    # train and evaluate SIP model
    attack_model = sip_model_train(env_args, sip_args, head_model, tokenizer, attack_model, optimizer, sip_train_loader, logger=logger)
    logger.info(f'SIP model training and evaluation done.')
    if not os.path.exists(sip_args.model_save_dir):
        os.makedirs(sip_args.model_save_dir)
    torch.save(attack_model, os.path.join(sip_args.model_save_dir, f'sip_{sip_args.model_name.split("/")[-1]}.pth'))
    logger.info(f'SIP model saved to {os.path.join(sip_args.model_save_dir, f"sip_{sip_args.model_name.split('/')[-1]}.pth")}')
    # evaluate SIP model
    evaluate_result = sip_model_evaluate(env_args, attack_model, head_model, tokenizer, sip_test_loader, logger=logger)
    logger.info(f'SIP model evaluation result: {evaluate_result}')
    pass
