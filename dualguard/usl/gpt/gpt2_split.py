from typing import List, Tuple, Optional, Union
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import GPT2Model,GPT2LMHeadModel,GPT2Config,GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast,BaseModelOutputWithPastAndCrossAttentions,CausalLMOutputWithCrossAttentions
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa,_prepare_4d_attention_mask_for_sdpa

from transformers.utils import (
    logging,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import get_peft_model, LoraConfig, TaskType
from functools import partial


from dualguard.defense.dp_noise import get_noise_multiplier
from dualguard.defense.dp_config import DP_EMBEDDING, DP_H2S_ACTIVATION, DP_T2S_GRADIENT, DPConfig
from dualguard.usl.split_config import SplitModelConfig
from dualguard.usl.split_model import SplitModel
logger = logging.get_logger(__name__)

class GPT2SplitModel(nn.Module):
    
    def __init__(self, gpt2:GPT2Model,split_config:SplitModelConfig,dp_config:DPConfig=DPConfig(),add_lora=False,lora_config:dict=None,*args, **kwargs):
        super().__init__()
        # print(f'dp_config: {dp_config}')
        self.config = gpt2.config
        self.dp_config=dp_config
        self.split_config = split_config
        self.head_model,self.server_model,self.tail_model=split_gpt(gpt2,split_config,dp_config)
        if add_lora:
            self.head_model=get_peft_model(self.head_model,lora_config)
            self.server_model=get_peft_model(self.server_model,lora_config)
            self.tail_model=get_peft_model(self.tail_model,lora_config)
        # self._register_grad_hooks()
        
    def enable_dp(self):
        self.dp_config.add_noise=True
        
    def disable_dp(self):
        self.dp_config.add_noise=False
             
    def reset_noise_multiplier(self,dataset_size:int,batch_size:int,epoch:int):
        if self.dp_config.epsilon != -1:
            self.dp_config.noise_factor  = get_noise_multiplier(self.dp_config.epsilon, self.dp_config.delta, batch_size=batch_size,
                                                dataset_size=dataset_size,epoch=epoch,local_dp=self.dp_config.local_dp,
                                                noise_type=self.dp_config.noise_type)
        else:
            self.dp_config.noise_factor = 0
            return
        
    def get_memory_size(self):
        param_size_on_cpu=0
        param_size_on_gpu=0
        for name,param in self.named_parameters():
            if param.requires_grad:
                if param.device.type=='cpu':
                    param_size_on_cpu+=param.numel()*param.dtype.itemsize
                elif param.device.type=='cuda':
                    param_size_on_gpu+=param.numel()*param.dtype.itemsize
                else:
                    print(f'unknown device type')
        return param_size_on_cpu,param_size_on_gpu
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,#代码目前有小bug，这里目前建议use_cache=False，否则可能影响推理
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        # return_dict: Optional[bool] = None,tail 输出就是dict
        cache_position: Optional[torch.LongTensor] = None,
        with_server: Optional[bool] = None,
        ll_mask: Optional[torch.Tensor] = None,
    ) -> CausalLMOutputWithCrossAttentions:
        temp=self.head_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
        )
        # print(temp[0])
        hidden_states,presents,past_key_values,attention_mask,head_mask,encoder_hidden_states,encoder_attention_mask,use_cache,\
        output_attentions,output_hidden_states,all_self_attentions,all_hidden_states,all_cross_attentions=temp
        if with_server and self.split_config.with_server:
            temp=self.server_model(hidden_status_from_head=hidden_states,
                                                    presents=presents,
                                                    past_key_values=past_key_values,
                                                    attention_mask=attention_mask,
                                                    head_mask=head_mask,
                                                    encoder_hidden_states=encoder_hidden_states,
                                                    encoder_attention_mask=encoder_attention_mask,
                                                    use_cache=use_cache,
                                                    output_attentions=output_attentions,
                                                    output_hidden_states=output_hidden_states,
                                                    all_self_attentions=all_self_attentions,
                                                    all_cross_attentions=all_cross_attentions,
                                                    all_hidden_states=all_hidden_states)
        hidden_states,presents,past_key_values,attention_mask,head_mask,encoder_hidden_states,encoder_attention_mask,use_cache,\
        output_attentions,output_hidden_states,all_self_attentions,all_hidden_states,all_cross_attentions=temp
        output=self.tail_model(hidden_status_from_server=hidden_states,
                                                presents=presents,
                                                past_key_values=past_key_values,
                                                attention_mask=attention_mask,
                                                head_mask=head_mask,
                                                encoder_hidden_states=encoder_hidden_states,
                                                encoder_attention_mask=encoder_attention_mask,
                                                use_cache=use_cache,
                                                output_attentions=output_attentions,
                                                output_hidden_states=output_hidden_states,
                                                all_self_attentions=all_self_attentions,
                                                all_hidden_states=all_hidden_states,
                                                all_cross_attentions=all_cross_attentions,
                                                labels=labels,ll_mask=ll_mask)
        return output
        pass
    
class GPT2Head(SplitModel):
    
    def __init__(self, config: GPT2Config,split_config:SplitModelConfig,dp_config:DPConfig):
        super().__init__(config,split_config,dp_config)
        if split_config.head_layer_num <= 0:
            logger.warning_once(
                'there is no head layer in the model, please check the split_config.head_layer_num ,\
                at least greater than 0'
            )
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.gradient_checkpointing = False
        self._attn_implementation = config._attn_implementation
        
    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _load_weight_from_pretrained_model_logically(self, pretrained_model:GPT2LMHeadModel,from_l,to_l):
        self.embed_dim = pretrained_model.config.hidden_size
        self.wte = pretrained_model.transformer.wte
        self.wpe = pretrained_model.transformer.wpe
        self.drop = pretrained_model.transformer.drop
        hidden_layers=pretrained_model.transformer.h
        hidden_layers:List[GPT2Block]
        self.layers = nn.ModuleList()
        for i in range(from_l,to_l):
            self.layers.append(hidden_layers[i])
        # self.layers[0].register_full_backward_hook(self.hook_fn)
            
    # def hook_fn(self, module, grad_input:Tuple[torch.Tensor], grad_output:Tuple[torch.Tensor]):
    #     print(" head model Inside Hook Function")
    #     # print("Gradient Input:", grad_input[0][0,0,:10])
    #     print("Gradient Output:", grad_output[0][0,0,:10])
        # return input
        

    def _load_weight_from_pretrained_model_physically(self, pretrained_model:GPT2LMHeadModel,from_l,to_l):
        self.embed_dim = pretrained_model.config.hidden_size
        self.wte=nn.Embedding(self.vocab_size,self.embed_dim)
        self.wpe=nn.Embedding(self.config.max_position_embeddings,self.embed_dim)
        self.drop = nn.Dropout(self.config.embd_pdrop)
        hidden_layers=pretrained_model.transformer.h
        hidden_layers:List[GPT2Block]
        self.layers = nn.ModuleList(
            [GPT2Block(self.config, layer_idx) for layer_idx in range(to_l-from_l)]
        )
        self.wte.load_state_dict(pretrained_model.transformer.wte.state_dict())
        self.wpe.load_state_dict(pretrained_model.transformer.wpe.state_dict())
        self.drop.load_state_dict(pretrained_model.transformer.drop.state_dict())
        for i in range(from_l,to_l):
            self.layers[i-from_l].load_state_dict(hidden_layers[i].state_dict())
        
    def load_from_pretrained_model(self, pretrained_model:GPT2LMHeadModel,logical=True):
        from_l=0
        to_l=self.split_config.head_layer_num
        if logical:
            self._load_weight_from_pretrained_model_logically(pretrained_model,from_l,to_l)
        else:
            self._load_weight_from_pretrained_model_physically(pretrained_model,from_l,to_l)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.config.n_layer) # 防止报错
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        # 输入embedding 加噪声 dxp
        if DP_EMBEDDING in self.dp_config.noise_positions:#在embedding层加入噪声
            # print(f'add noise to embedding')
            inputs_embeds = self.dxp(inputs_embeds,self.wte)
            
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # Attention mask.
        _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
        attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
        if self._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif _use_sdpa:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask=attention_mask,
                input_shape=(batch_size, input_shape[-1]),
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_length,
            )
        else:
            if attention_mask is not None:
                attention_mask = attention_mask[:, None, None, :]
                attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
            elif not self._attn_implementation == "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        # output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)
        # print(f'output_shape: {output_shape}')

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i in range(len(self.layers)):
            block = self.layers[i]
            layer_past = past_key_values[i] if past_key_values is not None else None
            # block = self.layers[i]
            # Model parallel
            # if self.model_parallel:
            #     torch.cuda.set_device(hidden_states.device)
            #     # Ensure layer_past is on same device as hidden_states (might not be correct)
            #     if layer_past is not None:
            #         layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
            #     # Ensure that attention_mask is always on the same device as hidden_states
            #     if attention_mask is not None:
            #         attention_mask = attention_mask.to(hidden_states.device)
            #     if isinstance(head_mask, torch.Tensor):
            #         head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            # print(f'head model hidden_states of layer {i}: {hidden_states[0,0,:10]}')
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)
        if DP_H2S_ACTIVATION in self.dp_config.noise_positions:#在激活层加入噪声
            # print(f'add noise to activation H2S')
            hidden_states = self.smashed_data_dp(hidden_states,sensitivity=8.0) 
            # # Model Parallel: If it's the last layer for that device, put things on the next device
            # if self.model_parallel:
            #     for k, v in self.device_map.items():
            #         if i == v[-1] and "cuda:" + str(k) != self.last_device:
            #             hidden_states = hidden_states.to("cuda:" + str(k + 1))
        return hidden_states,presents,past_key_values,attention_mask,head_mask,\
            encoder_hidden_states,encoder_attention_mask,use_cache,\
            output_attentions,output_hidden_states,all_self_attentions,all_hidden_states,all_cross_attentions

class GPT2Server(SplitModel): 
    
    def __init__(self, config: GPT2Config,split_config:SplitModelConfig,dp_config:DPConfig):
        super().__init__(config,split_config,dp_config)
        self.gradient_checkpointing = False
        if split_config.server_layer_num <= 0:
            logger.warning_once(
                'there is no server layer in the model, please check the split_config.server_layer_num ,\
                at least greater than 0'
            )
     
    def _load_weight_from_pretrained_model_logically(self, pretrained_model:GPT2LMHeadModel,from_l,to_l):
        hidden_layers=pretrained_model.transformer.h
        hidden_layers:List[GPT2Block]
        self.layers = nn.ModuleList()
        for i in range(from_l,to_l):
            self.layers.append(hidden_layers[i])

    def _load_weight_from_pretrained_model_physically(self, pretrained_model:GPT2LMHeadModel,from_l,to_l):
        hidden_layers=pretrained_model.transformer.h
        hidden_layers:List[GPT2Block]
        self.layers = nn.ModuleList(
            [GPT2Block(self.config, layer_idx+from_l) for layer_idx in range(to_l-from_l)]
        )
        for i in range(from_l,to_l):
            self.layers[i-from_l].load_state_dict(hidden_layers[i].state_dict())
        
    def load_from_pretrained_model(self, pretrained_model:GPT2LMHeadModel,logical=True):
        from_l=self.split_config.head_layer_num
        to_l=self.split_config.head_layer_num+self.split_config.server_layer_num
        if logical:
            self._load_weight_from_pretrained_model_logically(pretrained_model,from_l,to_l)
        else:
            self._load_weight_from_pretrained_model_physically(pretrained_model,from_l,to_l)
            
    def forward(self,
                hidden_status_from_head: torch.Tensor,
                presents: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None,
                use_cache: Optional[bool] = False,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                all_self_attentions: Optional[Tuple[torch.FloatTensor]] = None,
                all_hidden_states: Optional[Tuple[torch.FloatTensor]] = None,
                all_cross_attentions: Optional[Tuple[torch.FloatTensor]] = None,
                **kwargs
                ):
        # if past_key_values is None:
        #     past_length = 0
        #     past_key_values = tuple([None] * len(self.layers))
        # else:
        #     past_length = past_key_values[0][0].size(-2)
        if head_mask is None:
            head_mask = [None] * self.config.n_layer
        hidden_states=hidden_status_from_head
        for i in range(len(self.layers)):
            block = self.layers[i]
            layer_past = past_key_values[i] if past_key_values is not None else None
            # block=self.layers[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            # print(f'server model hidden_states of layer {i}: {hidden_states[0,0,:10]}')
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)
        return hidden_states,presents,past_key_values,attention_mask,head_mask,\
            encoder_hidden_states,encoder_attention_mask,use_cache,\
            output_attentions,output_hidden_states,all_self_attentions,all_hidden_states,all_cross_attentions
        
    
class GPT2Tail(SplitModel):
    
    def __init__(self, config: GPT2Config,split_config:SplitModelConfig,dp_config:DPConfig):
        super().__init__(config,split_config,dp_config)
        self.gradient_checkpointing = False
        if split_config.tail_layer_num <= 0:
            logger.warning_once(
                'there is no tail layer in the model, please check the split_config.tail_layer_num ,\
                at least greater than 0'
            )
   
    def _load_weight_from_pretrained_model_logically(self, pretrained_model:GPT2LMHeadModel,from_l,to_l):
        self.embed_dim=pretrained_model.config.hidden_size
        hidden_layers=pretrained_model.transformer.h
        hidden_layers:List[GPT2Block]
        self.layers = nn.ModuleList()
        for i in range(from_l,to_l):
            self.layers.append(hidden_layers[i])
        self.ln_f=pretrained_model.transformer.ln_f
        self.lm_head=pretrained_model.lm_head

    def _load_weight_from_pretrained_model_physically(self, pretrained_model:GPT2LMHeadModel,from_l,to_l):
        self.embed_dim=pretrained_model.config.hidden_size
        hidden_layers=pretrained_model.transformer.h
        hidden_layers:List[GPT2Block]
        self.layers = nn.ModuleList(
            [GPT2Block(self.config, layer_idx+from_l) for layer_idx in range(to_l-from_l)]
        )
        for i in range(from_l,to_l):
            self.layers[i-from_l].load_state_dict(hidden_layers[i].state_dict())
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=pretrained_model.config.layer_norm_epsilon)
        self.ln_f.load_state_dict(pretrained_model.transformer.ln_f.state_dict())
        self.lm_head = nn.Linear(self.embed_dim, self.config.vocab_size, bias=False)
        self.lm_head.load_state_dict(pretrained_model.lm_head.state_dict())
        
    def load_from_pretrained_model(self, pretrained_model:GPT2LMHeadModel,logical=True):
        from_l=self.split_config.total_hidden_layers-self.split_config.tail_layer_num
        to_l=self.split_config.total_hidden_layers
        if logical:
            self._load_weight_from_pretrained_model_logically(pretrained_model,from_l,to_l)
        else:
            self._load_weight_from_pretrained_model_physically(pretrained_model,from_l,to_l)
        if DP_T2S_GRADIENT in self.dp_config.noise_positions:  
            self._add_nosie_on_grad_from_tail_to_server()

    @staticmethod
    def wrapper(module,grad_input:Tuple[torch.Tensor],grad_output:Tuple[torch.Tensor],add_noise=False,dp_func=None):
        if add_noise:
            # print(f'add noise to grad from tail to server')
            grad_input_with_noise=dp_func(grad_input[0],sensitivity=8.0,return_new_tesnor=True)
            return (grad_input_with_noise,)
        else:
            return grad_input
            
    def _add_nosie_on_grad_from_tail_to_server(self):  
        hook_fn = partial(GPT2Tail.wrapper, add_noise=self.dp_config.add_noise,dp_func=self.smashed_data_dp)
        self.grad_dp_hook=self.layers[0].register_full_backward_hook(hook_fn)
            
    def forward(self,
                hidden_status_from_server: torch.Tensor,
                presents: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                head_mask: Optional[torch.LongTensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None,
                use_cache: Optional[bool] = False,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                all_self_attentions: Optional[Tuple[torch.FloatTensor]] = None,
                all_hidden_states: Optional[Tuple[torch.FloatTensor]] = None,
                all_cross_attentions: Optional[Tuple[torch.FloatTensor]] = None,
                labels: Optional[torch.LongTensor] = None,
                lm_mask: Optional[torch.LongTensor] = None,
                **kwargs
                ):
        if head_mask is None:
            head_mask = [None] * self.config.n_layer
        # if past_key_values is None:
        #     past_length = 0
        #     past_key_values = tuple([None] * len(self.layers))
        # else:
        #     past_length = past_key_values[0][0].size(-2)
        hidden_states=hidden_status_from_server
        for i in range(len(self.layers)):
            block = self.layers[i]
            layer_past = past_key_values[i] if past_key_values is not None else None
            # block=self.layers[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            # print(f'tail model hidden_states of layer {i}: {hidden_states[0,0,:10]}')
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)
            #loss
        hidden_states = self.ln_f(hidden_states)
        # hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits=shift_logits.view(-1,shift_logits.size(-1))
            shift_labels=shift_labels.view(-1)
            if lm_mask is not None:
                shift_lm_mask=lm_mask[..., 1:].contiguous()
                shift_lm_mask=shift_lm_mask.view(-1)
                loss_fct = CrossEntropyLoss(reduction='none')
                loss = loss_fct(shift_logits, shift_labels)
                loss=loss*shift_lm_mask.float()
                loss=loss.sum()/shift_lm_mask.sum()
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_cross_attentions,
            cross_attentions=all_cross_attentions,
        )

def split_gpt(pretrained_model:GPT2LMHeadModel,split_config:SplitModelConfig,dp_config:DPConfig)->Tuple[GPT2Head,Optional[GPT2Server],GPT2Tail]:
    # self.split_config = split_config
    config=pretrained_model.config
    if split_config.server_layer_num <= 0:
        split_config.server_layer_num = config.n_layer - split_config.tail_layer_num-split_config.head_layer_num
    head_model=GPT2Head(config,split_config,dp_config)
    head_model.load_from_pretrained_model(pretrained_model,logical=split_config.logicl_load)
    # print(f'head model loaded scucessfully')
    server_model=None
    if split_config.with_server:
        server_model=GPT2Server(config,split_config,dp_config)
        server_model.load_from_pretrained_model(pretrained_model,logical=split_config.logicl_load)
    # print(f'server model loaded scucessfully')
    tail_model=GPT2Tail(config,split_config,dp_config)
    tail_model.load_from_pretrained_model(pretrained_model,logical=split_config.logicl_load)
    # print(f'tail model loaded scucessfully')
    return head_model,server_model,tail_model