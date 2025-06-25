from abc import ABC, abstractmethod

from transformers import PretrainedConfig, PreTrainedModel
from dualguard.defense.dp_config import DPConfig
from dualguard.usl.split_config import SplitModelConfig

import torch


class SplitModel(PreTrainedModel, ABC):

    def __init__(self, config: PretrainedConfig = None, split_config: SplitModelConfig = None, dp_config: DPConfig = None, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.split_config = split_config
        self.dp_config = dp_config

    # 用于从预训练模型中加载模型
    @abstractmethod
    def load_from_pretrained_model(self, *args, **kwargs):
        raise NotImplementedError('load_from_pretrained_model is not implemented')

    def disabled_dp(self, disable: bool = True):
        self.dp_config.add_noise = not disable

    def smashed_data_dp(self, smashed_data: torch.Tensor, sensitivity=8.0, return_new_tesnor=False):

        # copy from https://github.com/StupidTrees/SplitLLM/blob/benchmark/sfl/model/noise/fdp.py
        if self.dp_config.epsilon == 0 or not self.dp_config.add_noise:
            return smashed_data
        if return_new_tesnor:
            smashed_data = smashed_data.clone()
        batch_size, seq_len, hidden_size = smashed_data.size()
        # print(f'mean of smashed_data : {smashed_data.abs().mean()}')
        # clip the hidden_states by x=x/scale, scale=max(1, x.inf_norm/G), scale:(batch_size, 1, 1)
        G = 2000
        scale = torch.max(
            torch.norm(smashed_data.view(batch_size, -1), p=float('inf'), dim=1, keepdim=True) / G, torch.ones(batch_size, 1).to(smashed_data.device)
        )
        if smashed_data.dtype == torch.float16:
            scale = scale.half()
        smashed_data = (smashed_data.view(batch_size, -1) / scale).view(batch_size, seq_len, hidden_size)
        noise = torch.distributions.Laplace(0, smashed_data.abs().mean()).sample(smashed_data.size()).to(smashed_data.device)
        if smashed_data.dtype == torch.float16:
            noise = noise.half()
        return noise + smashed_data

    def dxp(self, inputs_embeds: torch.Tensor, embed_tokens: torch.nn.Embedding, **kwargs) -> torch.Tensor:
        if self.dp_config.epsilon == 0 or not self.dp_config.add_noise:
            return inputs_embeds
        if not hasattr(self, 'all_embeds') or self.all_embeds is None:
            all_words = torch.tensor(list([i for i in range(self.vocab_size)])).to(inputs_embeds.device)
            self.all_embeds = embed_tokens(all_words)
        self.all_embeds = self.all_embeds.to(inputs_embeds.device)
        with torch.no_grad():
            batch_size, seq_len, embed_size = inputs_embeds.shape
            # Sample noise from multivariate normal distribution
            cov_matrix = torch.eye(embed_size).expand(batch_size, seq_len, embed_size, embed_size)
            if not hasattr(self, 'normal_dist'):
                self.normal_dist = torch.distributions.MultivariateNormal(torch.zeros(embed_size), covariance_matrix=cov_matrix[0, 0])
            noise_v = self.normal_dist.sample(inputs_embeds.shape[:2])
            norm = torch.linalg.norm(noise_v, dim=-1, keepdim=True)
            norm = torch.where(norm > 0, norm, torch.ones_like(norm))
            noise_v = noise_v / norm
            # Sample scale from gamma distribution
            alpha = embed_size
            # self.scale = relative epsilon = epsilon/embed_size
            beta = self.dp_config.epsilon * embed_size
            if not hasattr(self, 'gamma_dist'):
                self.gamma_dist = torch.distributions.Gamma(torch.tensor([alpha]).float(), torch.tensor([beta]))
            scale = self.gamma_dist.sample()
            # Apply noise
            noise = scale * noise_v
            if inputs_embeds.dtype == torch.float16:
                noise = noise.half()
            inputs_embeds = inputs_embeds + noise.to(inputs_embeds.device)
        is_half = inputs_embeds.dtype == torch.float16 or self.all_embeds.dtype == torch.float16
        if is_half:
            inputs_embeds = inputs_embeds.float()
        cosine_similarities = torch.matmul(inputs_embeds, self.all_embeds.transpose(0, 1))
        max_token = torch.argmax(cosine_similarities, dim=-1)  # bs
        res = self.all_embeds[max_token]
        if is_half:
            res = res.half()
        return res  # bsh
