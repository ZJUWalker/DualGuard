from dataclasses import dataclass, field
from torch.optim import SGD, Adam, AdamW, Optimizer
import torch
from typing import Dict
from peft import get_peft_model, LoraConfig, TaskType
from dualguard.utils import env


@dataclass
class EnvArgs:
    device: str = field(default="cuda:0")
    random_seed: int = field(default=0)
    work_dir: str = field(default="xxx")  # it seems that this is not used


@dataclass
class LogArgs:
    log_dir: str = field(default=env.sip_log_dir)
    log_level: str = field(default="INFO")
    log_file_name: str = field(default="default.log")
    format: str = field(default="%(asctime)s %(levelname)s %(name)s %(message)s")
    mode: str = field(default="a")


@dataclass
class SIPTrainArgs:
    epochs: int = field(default=2)
    batch_size: int = field(default=4)
    model_name: str = field(default='gpt/gpt2-large')
    split_point: int = field(default=3)
    aux_dataset: str = field(default='wikitext')
    optimizer: torch.optim.Optimizer.__class__ = field(default=AdamW, metadata={'help': "the optimizer of warmup"})
    optimizer_kwargs: Dict = field(
        default_factory=lambda: {"lr": 1e-4, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01, "amsgrad": False},
        metadata={'help': "the optimizer kwargs of warmup"},
    )
    model_save_dir: str = field(default=env.sip_model_dir)


@dataclass
class SIPAttackArgs:
    sip_model_dir: str = field(default=env.sip_model_dir)
    head_model_dir: str = field(default=env.warmup_model_dir)
    epochs: int = field(default=3)
    batch_size: int = field(default=4)


@dataclass
class DatasetArgs:
    dataset_name: str = field(default="dialogsum")
    splits: list = field(default_factory=lambda: ["train", "test", "validation"])
    max_seq_len: int = field(default=512)
    batch_size: int = field(default=4)
    shuffle: bool = field(default=False)


'''
# model_name='Qwen/qwen2-1.5b'
# model_name='meta-llama/Llama-3.2-1B'
model_name='gpt/gpt2-large'
'''


@dataclass
class PretrainedModelArgs:
    model_name: str = field(default='gpt/gpt2-large', metadata={'help': "the name of the model"})


@dataclass
class AttackArgs:
    attack_name: str = field(default="tag", metadata={'help': "the name of the attack"})


@dataclass
class TAGAttackArgs:
    target_model_dir: str = field(default=env.usl_model_dir, metadata={'help': "the directory of target model after usl"})
    attack_iters: int = field(default=300, metadata={'help': "the number of attack iterations"})
    attack_optimizer: torch.optim.Optimizer.__class__ = field(default=AdamW, metadata={'help': "the optimizer of attack"})
    attack_optimizer_kwargs: Dict = field(
        default_factory=lambda: {"lr": 0.09, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0, "amsgrad": False},
        metadata={'help': "the optimizer kwargs of attack"},
    )
    beta: float = field(default=0.85, metadata={'help': "the beta of the attack loss"})
    sample_num: int = field(default=20, metadata={'help': "the sample num of the attack"})
    is_warmup_usl: bool = field(default=False, metadata={'help': "whether to use pretrained loss"})


@dataclass
class LampAttackArgs(TAGAttackArgs):
    attack_iters: int = field(default=200, metadata={'help': "the number of attack iterations"})
    lamp_interval: int = field(default=50, metadata={'help': "the interval of lamp training"})
    lamp_iters: int = field(default=30, metadata={'help': "the number of lamp iterations"})


@dataclass
class SMAAttackArgs(TAGAttackArgs):
    sma_iters: int = field(default=20, metadata={'help': "the number of attack iterations"})


@dataclass
class BISRAttackArgs(SMAAttackArgs):
    dlg_type: str = field(default="tag", metadata={'help': "the type of dialog"})
    pass


# model_name='Qwen/qwen2-1.5b'
# model_name='meta-llama/Llama-3.2-1B'
# model_name='gpt/gpt2-large'
@dataclass
class TrainArgs(PretrainedModelArgs):
    epochs: int = field(default=10, metadata={'help': "the number of warmup iterations"})
    optimizer: torch.optim.Optimizer.__class__ = field(default=Adam, metadata={'help': "the optimizer of warmup"})
    optimizer_kwargs: Dict = field(
        default_factory=lambda: {"lr": 1e-5, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0, "amsgrad": False},
        metadata={'help': "the optimizer kwargs of warmup"},
    )
    half: bool = field(default=False, metadata={'help': "whether to use fp16 precision"})
    log_interval: int = field(default=20, metadata={'help': "the interval of logging"})
    validation_an_epoch: int = field(default=5, metadata={'help': "the interval of validation"})
    log_file: str = field(default="training.log", metadata={'help': "the file of logging"})
    early_stop_patience: int = field(default=5, metadata={'help': "the number of evals without improvement"})
    early_stop_threshold: float = field(default=0.005, metadata={'help': "the threshold of improvement"})
    use_lora: bool = field(default=False, metadata={'help': "whether to use lora"})
    lora_config: LoraConfig = field(
        default_factory=lambda: LoraConfig(task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "v_proj"]),
        metadata={'help': "the config of lora"},
    )


@dataclass
class WarmupArgs(TrainArgs):
    warm_up_epochs: int = field(default=4, metadata={'help': "the number of warmup iterations"})
    sip_model_dir: str = field(default=env.sip_model_dir, metadata={'help': "the path of sip model"})
    save_dir: str = field(default=env.warmup_model_dir, metadata={'help': "the directory of usl model"})
    warm_up_optimizer: torch.optim.Optimizer.__class__ = field(default=Adam, metadata={'help': "the optimizer of warmup"})
    warm_up_optimizer_kwargs: Dict = field(
        default_factory=lambda: {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0, "amsgrad": False},
        metadata={'help': "the optimizer kwargs of warmup"},
    )
    intermediate_scale: int = field(default=2, metadata={'help': "the scale of intermeidate model"})
    loss_weights: torch.Tensor = field(
        default=torch.nn.Parameter(torch.Tensor([1.0, 2.0, 3.0]), requires_grad=True), metadata={'help': "the loss weights of warmup model"}
    )
    pt_loss_w: float = field(default=70.0, metadata={'help': "the loss weight of pt model"})
    attack_loss_w: float = field(default=30.0, metadata={'help': "the loss weight of attack model"})


@dataclass
class USLTrainArgs(TrainArgs):
    use_naive_usl: bool = field(default=True, metadata={'help': "whether to use naive usl"})
    wp_model_dir: str = field(default=env.warmup_model_dir, metadata={'help': "the directory of usl model"})
    save_dir: str = field(default=env.usl_model_dir, metadata={'help': "the directory of usl model"})
    frozen_head: bool = field(default=False, metadata={'help': "whether to freeze the head of usl model"})
    frozen_server: bool = field(default=False, metadata={'help': "whether to freeze the server of usl model"})
    frozen_tail: bool = field(default=False, metadata={'help': "whether to freeze the tail of usl model"})
    with_pt_tail_model: bool = field(default=False, metadata={'help': "whether to use pretrained tail model"})
    split_point: int = field(default=2, metadata={'help': "the split point of sip model loss"})
    lambda_3: float = field(default=10.0, metadata={'help': "the lambda3 of pretrained tail model loss"})


@dataclass
# haven't use this class yet
class MeltExperimentArgs(TrainArgs):
    warm_up_epochs: int = field(default=4, metadata={'help': "the number of warmup iterations"})
    sip_model_dir: str = field(default=env.sip_model_dir, metadata={'help': "the path of sip model"})
    warm_up_optimizer: torch.optim.Optimizer.__class__ = field(default=Adam, metadata={'help': "the optimizer of warmup"})
    warm_up_optimizer_kwargs: Dict = field(
        default_factory=lambda: {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0, "amsgrad": False},
        metadata={'help': "the optimizer kwargs of warmup"},
    )
    intermediate_scale: int = field(default=2, metadata={'help': "the scale of intermeidate model"})
    save_dir: str = field(default="./", metadata={'help': "the directory of usl model"})
    lambda_params: tuple = field(default_factory=lambda: (1, 30, 70, 70), metadata={'help': "the lambda params of melt"})
