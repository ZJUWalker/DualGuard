from dataclasses import dataclass, field
from typing import List, Tuple, Union,Optional,Dict

DP_EMBEDDING = "DP_EMBEDDING"
DP_H2S_ACTIVATION = "DP_H2S_ACTIVATION"
DP_T2S_GRADIENT = "DP_T2S_GRADIENT"


@dataclass
class DPConfig(object):
    noise_positions: List[str] = field(
        default_factory=list,
        metadata={"help": "dp noise parameter: noise_positions"}
    )
    epsilon: float = field(
        default=4.0,
        metadata={"help": "dp noise parameter: epsilon"}
    )

    epsilon2: float = field(
        default=2.0,
        metadata={"help": "dp noise parameter: epsilon"}
    )

    delta: float = field(
        default=1e-5,
        metadata={"help": "dp noise parameter: delta"}
    )

    sampling_prob: float = field(
        default=1.0,
        metadata={"help": "dp noise parameter: sampling_prob"}
    )

    norm_c: float = field(
        default=1.0,
        metadata={"help": "dp noise parameter: norm_c"}
    )

    noise_type: str = field(
        default="GM",
        metadata={"help": "dp noise parameter: noise_type"}
    )

    noise_layer: int = field(
        default=-1,
        metadata={"help": "which layer to add noise when the noise position is encoder"}
    )

    encoder_sub_noise_position : str = field(
        default="",
        metadata={"help": "dp noise parameter: encoder_sub_noise_position"}
    )

    proj_dim: int = field(
        default=-1,
        metadata={"help": "dim of projection layer"}
    )

    label_dp: bool = field(
        default=False,
        metadata={"help": "whether use label dp"}
    )

    local_dp: bool = field(
        default=False,
        metadata={"help": "whether to use local dp"}
    )

    add_noise_inference: bool = field(
        default=False, metadata={"help": "Whether to add noise during inference phase"}
    )

    freeze_pre_noise_layers: bool = field(
        default=False, metadata={"help": "Whether to freeze pre-noise layer parameters"}
    )
    
    add_noise: bool = field(
        default=True, metadata={"help": "Whether to add noise during training phase"}
    )