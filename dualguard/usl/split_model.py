from abc import ABC, abstractmethod

from transformers import PretrainedConfig,PreTrainedModel
from dualguard.defense.dp_config import DPConfig
from usl.split_config import SplitModelConfig


import torch

class SplitModel(PreTrainedModel,ABC):

    def __init__(self,config:PretrainedConfig=None,split_config:SplitModelConfig=None,dp_config:DPConfig=None,*args,**kwargs):
        super().__init__(config,*args,**kwargs)
        self.config = config
        self.split_config = split_config
        self.dp_config = dp_config

    #用于从预训练模型中加载模型
    @abstractmethod
    def load_from_pretrained_model(self,*args,**kwargs):
        raise NotImplementedError('load_from_pretrained_model is not implemented')

    #用于将模型深度拷贝一份
    # @abstractmethod
    # def deepcopy(self):
    #     raise NotImplementedError('deepcopy is not implemented')
    
    def disabled_dp(self,disable:bool=True):
        self.dp_config.add_noise = not disable
        
        
    def smashed_data_dp(self,smashed_data:torch.Tensor, sensitivity=8.0,return_new_tesnor=False):
        
        # copy from https://github.com/StupidTrees/SplitLLM/blob/benchmark/sfl/model/noise/fdp.py
        if self.dp_config.epsilon == 0 or not self.dp_config.add_noise:
            return smashed_data
        if return_new_tesnor:
            smashed_data = smashed_data.clone()
        batch_size, seq_len, hidden_size = smashed_data.size()
        # print(f'mean of smashed_data : {smashed_data.abs().mean()}')
        # clip the hidden_states by x=x/scale, scale=max(1, x.inf_norm/G), scale:(batch_size, 1, 1)
        G = 2000
        scale = torch.max(torch.norm(smashed_data.view(batch_size, -1), p=float('inf'), dim=1, keepdim=True) / G,
                          torch.ones(batch_size, 1).to(smashed_data.device))
        if smashed_data.dtype == torch.float16:
            scale = scale.half()
        smashed_data = (smashed_data.view(batch_size, -1) / scale).view(batch_size, seq_len, hidden_size)
        noise = torch.distributions.Laplace(0, sensitivity / self.dp_config.epsilon).sample(smashed_data.size()).to(smashed_data.device)
        if smashed_data.dtype == torch.float16:
            noise = noise.half()
        # new_smashed_data = smashed_data + noise
        # print(f'new_smashed_grad : {new_smashed_data[0,0,:]}')
        # return new_smashed_data
        return noise+smashed_data
        # epsilon=self.dp_config.epsilon
        # clipping_threshold=self.dp_config.norm_c
        # sensitivity = self._get_sensitivity(smashed_data,clipping_threshold=clipping_threshold)

        # # print(f'original sensitivity : {smashed_data.abs().max(dim=-1,keepdim=True)}')
        # with torch.no_grad():
        # # Step 1: Clipping
        #     norm = torch.norm(smashed_data, p=float('inf'), dim=-1, keepdim=True)  # L∞ norm
        #     # print(f'norm : {norm[0,2,0]}')
        #     scaler=torch.maximum(torch.tensor(1.0), norm / clipping_threshold)
        #     # print(f'scaler : {scaler[0,2,0]}')
        #     clipped_data = smashed_data / scaler
        #     # print(f'clipped_data  : {clipped_data[0,2,100:200]}')
        #     # Step 2: Calculate Laplace noise scale
        #     noise_scale = sensitivity / epsilon  # Laplace scale = Δf_btm / ε
        #     # print(f'noise_scale : {noise_scale}')
        #     # if not hasattr(self,'laplace_dist'):
        #     laplace_dist = torch.distributions.Laplace(loc=0.0, scale=noise_scale)
        #     # Step 3: Sample noise from Laplace distribution
        #     noise = laplace_dist.sample(clipped_data.shape).to(smashed_data.device)
        #     # print(f'noise : {noise[0,2,:100]}')
        #     # Step 4: Add noise to clipped data
        # dp_protected_data = clipped_data + noise
        # return dp_protected_data
    
    def dxp(self, inputs_embeds:torch.Tensor,embed_tokens:torch.nn.Embedding, **kwargs)->torch.Tensor:
        if self.dp_config.epsilon == 0 or not self.dp_config.add_noise:
            return inputs_embeds
        if not hasattr(self,'all_embeds') or self.all_embeds is None:
            all_words = torch.tensor(list([i for i in range(self.vocab_size)])).to(inputs_embeds.device)
            self.all_embeds = embed_tokens(all_words)
        self.all_embeds=self.all_embeds.to(inputs_embeds.device)
        with torch.no_grad():
            batch_size, seq_len, embed_size = inputs_embeds.shape
            # Sample noise from multivariate normal distribution
            cov_matrix = torch.eye(embed_size).expand(batch_size, seq_len, embed_size, embed_size)
            if not hasattr(self,'normal_dist'):
                self.normal_dist = torch.distributions.MultivariateNormal(torch.zeros(embed_size), covariance_matrix=cov_matrix[0, 0])
            noise_v = self.normal_dist.sample(inputs_embeds.shape[:2])
            norm = torch.linalg.norm(noise_v, dim=-1, keepdim=True)
            norm = torch.where(norm > 0, norm, torch.ones_like(norm))
            noise_v = noise_v / norm
            # Sample scale from gamma distribution
            alpha = embed_size
            # self.scale = relative epsilon = epsilon/embed_size
            beta = self.dp_config.epsilon * embed_size
            if not hasattr(self,'gamma_dist'):
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
        max_token = torch.argmax(cosine_similarities, dim=-1)# bs 
        res = self.all_embeds[max_token]
        if is_half:
            res = res.half()
        return res #bsh
    
# #用于标识模型的wrapper
# class SplitModelWrapper(object):
    
#     def __init__(self):
#         pass
    


