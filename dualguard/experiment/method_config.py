from dataclasses import dataclass
from dualguard.defense.dp_config import DPConfig
from dualguard.utils.configs import USLTrainArgs
from peft import LoraConfig, TaskType


'''
注意事项：
模型保存地址为*/static下
sip模型保存格式为 /static/sip/sip_{model_name}.pth
其中warmup后的模型保存格式为 /static/warmup/{model_name}/{dataset_name}/{[head.pth|tail.pth]}
usl后的模型保存格式为 /static/usl/{model_name}/{dataset_name}/{method}{noise_scaler if add_noise else ''}{[head.pth|server.pth|tail.pth]}
'''
# 模型
GPT2_LARGE = 'gpt/gpt2-large'
QWEN2 = 'Qwen/qwen2-1.5b'
LLAMA3 = 'meta-llama/Llama-3.2-1B'

# 数据集
GSM8K = 'gsm8k'
CODEALPACA = 'codealpaca'
DIALOGSUM = 'dialogsum'
E2E = 'e2e'
PIQA = 'piqa'


@dataclass
class MethodConfig:
    info: str
    prefix: str
    usl_args: USLTrainArgs
    dp_config: DPConfig


naive_config = MethodConfig(
    info='train/eval with naive usl',
    prefix='naive_usl',
    usl_args=USLTrainArgs(frozen_head=False, use_lora=True, use_naive_usl=True, with_pt_tail_model=False),
    dp_config=DPConfig(add_noise=False),
)
dp_forward_config = MethodConfig(
    info='train/eval with dp-forward',
    prefix='dp_forward',
    usl_args=USLTrainArgs(frozen_head=False, use_lora=True, use_naive_usl=True, with_pt_tail_model=False),
    dp_config=DPConfig(add_noise=True, noise_positions=['DP_H2S_ACTIVATION'], epsilon=2.0),
)
dxp_config = MethodConfig(
    info='train/eval with dxp',
    prefix='dxp',
    usl_args=USLTrainArgs(frozen_head=False, use_lora=True, use_naive_usl=True, with_pt_tail_model=False),
    dp_config=DPConfig(add_noise=True, noise_positions=['DP_EMBEDDING'], epsilon=2.0),
)
dp_sgd_config = MethodConfig(
    info='train/eval with dp-sgd',
    prefix='dp_sgd',
    usl_args=USLTrainArgs(frozen_head=False, use_lora=True, use_naive_usl=True, with_pt_tail_model=False),
    dp_config=DPConfig(add_noise=True, noise_positions=['DP_T2S_GRADIENT'], epsilon=2.0),
)
dualguard_config = MethodConfig(
    info='train/eval with dualguard',
    prefix='dualguard',
    usl_args=USLTrainArgs(frozen_head=True, use_lora=True, use_naive_usl=False, with_pt_tail_model=True),
    dp_config=DPConfig(add_noise=False),
)

gpt_lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=32, lora_dropout=0.1, target_modules=["attn.c_proj", "attn.c_attn"])

qwen_llama_lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=32, lora_dropout=0.1, target_modules=['q_proj', 'v_proj'])
