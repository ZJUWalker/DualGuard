import argparse
from utils.gpu import add_gpu_params
from utils.optimizer import (
    add_optimizer_params,
)
from typing import Dict, Any
from utils.load_models import list_models


#需要大修 TODO
def args_parser()->Dict[str, Any]:
    parser = argparse.ArgumentParser(description="PyTorch GPT2 ft script")

    add_gpu_params(parser)
    add_optimizer_params(parser)
    add_dp_args_parser(parser)
    
    parser.add_argument("--train_data0", default="./data/dart/train_formatted.jsonl", help="location of training data corpus")

    parser.add_argument( "--train_data1", default="./data/e2e/train1.jsonl", help="location of training data corpus")

    parser.add_argument("--train_data2", default="./data/e2e/train2.jsonl", help="location of training data corpus")

    parser.add_argument("--valid_data", default="./data/e2e/valid.jsonl", help="location of validation data corpus")

    parser.add_argument("--train_batch_size", type=int, default=4, help="training batch size")

    parser.add_argument("--valid_batch_size", type=int, default=4, help="validation batch size")

    parser.add_argument("--grad_acc", type=int, default=1, help="gradient accumulation steps")
    
    parser.add_argument("--epoch", type=int, default=10, help="training epoch")

    parser.add_argument("--clip", type=float, default=0.0, help="gradient clip")

    parser.add_argument("--seq_len", type=int, default=512, help="number of tokens to predict.")

    parser.add_argument("--model_card", default="meta-llama/Llama-3.2-1B", choices=list_models(), help="model names",)

    parser.add_argument("--init_checkpoint", default="/home/lzh/projects/privacy-split-llm/experiment/pretrained_checkpoints/gpt2-large-pytorch_model.bin", help="pretrained checkpoint path")

    parser.add_argument("--fp16", action="store_true", help="train model with fp16")

    parser.add_argument("--log_interval", type=int, default=100, help="log interval")

    parser.add_argument("--warmup_log_interval", type=int, default=100, help="log interval")

    parser.add_argument("--warmup_eval_interval", type=int, default=1000, help="log interval")

    parser.add_argument("--eval_interval", type=int, default=1000, help="eval interval")

    parser.add_argument("--save_interval", type=int, default=400000, help="save interval")

    parser.add_argument("--work_dir", type=str, default="./trained_models/GPT2_M/", help="working folder.")

    parser.add_argument("--lora_dim", type=int, default=2, help="lora attn dimension")

    parser.add_argument("--lora_alpha", type=int, default=32, help="lora attn alpha")

    parser.add_argument(
        "--obj",
        default="clm",
        choices=["jlm", "clm"],
        help="language model training objective",
    )

    parser.add_argument(
        "--lora_dropout",
        default=0.0,
        type=float,
        help="dropout probability for lora layers",
    )

    parser.add_argument("--num_clients", default=1, type=float, help="number of clients")

    parser.add_argument("--label_smooth", default=0, type=float, help="label smoothing")

    parser.add_argument("--roll_interval", type=int, default=-1, help="rolling interval")

    parser.add_argument("--roll_lr", type=float, default=0.00001, help="rolling learning rate")

    parser.add_argument("--roll_step", type=int, default=100, help="rolling step")

    parser.add_argument("--eval_epoch", type=int, default=1, help="eval per number of epochs")

    parser.add_argument("--is_ALM", type=int, default=True, help="is Autoregressive Language Modeling or not")

    args = parser.parse_args()
    return args

def add_dp_args_parser(parser)->Dict[str, Any]:
    # parser = argparse.ArgumentParser(description="PyTorch GPT2 ft script")
    parser.add_argument("--defence_method",type=str, default="dp_forward", help="location of training data corpus")
    return parser.parse_args()

def SIP_train_args_parser()->Dict[str, Any]:
    parser = argparse.ArgumentParser(description="inversion script")

    parser.add_argument( "--opt_cls", default="AdamW", help="location of training data corpus")
    parser.add_argument( "--test_frac", default=0.1, help="location of training data corpus")
    parser.add_argument( "--lr", default=0.001, help="location of training data corpus")
    parser.add_argument( "--weight_decay", default=1e-05, help="location of training data corpus")
    parser.add_argument( "--epochs", default=4, help="location of training data corpus")
    parser.add_argument( "--gating_epochs", default=15, help="location of training data corpus")
    parser.add_argument( "--ft_epochs", default=4, help="location of training data corpus")
    parser.add_argument( "--batch_size", default=8, help="location of training data corpus")
    parser.add_argument( "--log_to_wandb", default=False, help="location of training data corpus")
    parser.add_argument( "--save_checkpoint", default=True, help="location of training data corpus")
    parser.add_argument( "--checkpoint_freq", default=5, help="location of training data corpus")
    parser.add_argument( "--save_threshold", default=0.1, help="location of training data corpus")

    train_args = parser.parse_args()
    return train_args

def warmup_args_parser()->Dict[str, Any]:
    parser = argparse.ArgumentParser(description="inversion script")

    parser.add_argument( "--opt_cls", default="AdamW", help="location of training data corpus")
    parser.add_argument( "--test_frac", default=0.1, help="location of training data corpus")
    parser.add_argument( "--lr", default=0.001, help="location of training data corpus")
    parser.add_argument( "--weight_decay", default=1e-05, help="location of training data corpus")
    parser.add_argument( "--epochs", default=5, help="location of training data corpus")
    parser.add_argument( "--gating_epochs", default=15, help="location of training data corpus")
    parser.add_argument( "--ft_epochs", default=4, help="location of training data corpus")
    parser.add_argument( "--batch_size", default=6, help="location of training data corpus")
    parser.add_argument( "--log_to_wandb", default=False, help="location of training data corpus")
    parser.add_argument( "--save_checkpoint", default=True, help="location of training data corpus")
    parser.add_argument( "--checkpoint_freq", default=5, help="location of training data corpus")
    parser.add_argument( "--save_threshold", default=0.1, help="location of training data corpus")

    train_args = parser.parse_args()
    return train_args