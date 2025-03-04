import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, LayerNorm, MSELoss, BCEWithLogitsLoss
from torch.nn.utils import skip_init
from typing import Optional, Tuple, Union, List, Callable, Dict, Any
from copy import deepcopy

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig, ModelOutput

from usl.chatglm.configuration_chatglm import ChatGLMConfig
# from .split_config import SplitModelConfig
from usl.split_config import SplitModelConfig
from usl.chatglm.chatglm_model import ChatGLMForConditionalGeneration,ChatGLMPreTrainedModel,ChatGLMModel,Embedding,LayerNorm,PrefixEncoder\
    ,RMSNorm,RotaryEmbedding,GLMBlock,GLMTransformer
from typing import List, Tuple,Dict,Optional,Union

logger = logging.get_logger(__name__)

def default_init(cls, *args, **kwargs):
    return cls(*args, **kwargs)

# TODO 实现glm split模型的封装
class ChatGLMSplitModel(nn.Module):
    
    pass

#修改Transerformer
class HeadChatGLMTransformer(nn.Module):
    """Splitted Transformer class."""

    def __init__(self, glm_config: ChatGLMConfig,split_config:SplitModelConfig, device=None):
        super(HeadChatGLMTransformer, self).__init__()
        self.split_config=split_config
        self.fp32_residual_connection = glm_config.fp32_residual_connection
        # self.post_layer_norm = glm_config.post_layer_norm # head 部分不存在layernorm

        # Number of layers in head model.
        self.num_layers = split_config.head_layer_num

        # Transformer layers.
        def build_layer(layer_number):
            return GLMBlock(glm_config, layer_number, device=device)

        self.layers = torch.nn.ModuleList([build_layer(i + 1) for i in range(self.num_layers)])

        #head 部分不存在layernorm
        # if self.post_layer_norm:
        #     LayerNormFunc = RMSNorm if glm_config.rmsnorm else LayerNorm
        #     # Final layer norm before output.
        #     self.final_layernorm = LayerNormFunc(glm_config.hidden_size, eps=glm_config.layernorm_epsilon, device=device,
        #                                          dtype=glm_config.torch_dtype)

        self.gradient_checkpointing = False

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_caches=None,
            use_cache: Optional[bool] = True,
            output_hidden_states: Optional[bool] = False,
    ):
        if not kv_caches:
            kv_caches = [None for _ in range(self.num_layers)]
        presents = () if use_cache else None
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_self_attentions = None
        all_hidden_states = () if output_hidden_states else None
        for index in range(self.num_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer = self._get_layer(index)
            if self.gradient_checkpointing and self.training:
                layer_ret = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_caches[index],
                    use_cache
                )
            else:
                layer_ret = layer(
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_cache=kv_caches[index],
                    use_cache=use_cache
                )
            hidden_states, kv_cache = layer_ret
            if use_cache:
                presents = presents + (kv_cache,)

        # if output_hidden_states:
        #     all_hidden_states = all_hidden_states + (hidden_states,)

        # # Final layer norm.
        # if self.post_layer_norm:
        #     hidden_states = self.final_layernorm(hidden_states)

        return hidden_states,attention_mask,rotary_pos_emb,\
            presents,output_hidden_states,use_cache, all_hidden_states, all_self_attentions

    def load_from_pretrained(self, glm_model:ChatGLMModel):
        # embedding=glm_model._modules['embedding']
        # rotary_pos_emb=glm_model._modules['rotary_pos_emb']
        glm_blocks=glm_model._modules['encoder']._modules['layers']
        for i in range(self.num_layers):
            self.layers[i]=glm_blocks[i]
        pass
    
class ServerChatGLMTransformer(nn.Module):
    """Splitted Transformer class."""


    def __init__(self, glm_config: ChatGLMConfig,split_config:SplitModelConfig, device=None):
        super(ServerChatGLMTransformer, self).__init__()
        self.split_config=split_config
        self.fp32_residual_connection = glm_config.fp32_residual_connection
        # self.post_layer_norm = glm_config.post_layer_norm # server 部分不存在layernorm

        # Number of layers in head model.
        self.num_layers = split_config.server_layer_num

        # Transformer layers.
        def build_layer(layer_number):
            return GLMBlock(glm_config, layer_number, device=device)

        self.layers = torch.nn.ModuleList([build_layer(i + 1) for i in range(self.num_layers)])

        #head 部分不存在layernorm
        # if self.post_layer_norm:
        #     LayerNormFunc = RMSNorm if glm_config.rmsnorm else LayerNorm
        #     # Final layer norm before output.
        #     self.final_layernorm = LayerNormFunc(glm_config.hidden_size, eps=glm_config.layernorm_epsilon, device=device,
        #                                          dtype=glm_config.torch_dtype)

        self.gradient_checkpointing = False

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def forward(
            self,
            hidden_states, 
            attention_mask,
            rotary_pos_emb,
            presents,
            output_hidden_states,
            use_cache, 
            all_hidden_states, 
            all_self_attentions
    ):
        kv_caches = [None for _ in range(self.num_layers)]
        for index in range(self.num_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer = self._get_layer(index)
            if self.gradient_checkpointing and self.training:
                layer_ret = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_caches[index],
                    use_cache
                )
            else:
                layer_ret = layer(
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_cache=kv_caches[index],
                    use_cache=use_cache
                )
            hidden_states, kv_cache = layer_ret
            if use_cache:
                presents = presents + (kv_cache,)

        return hidden_states,attention_mask,rotary_pos_emb,\
            presents,output_hidden_states,use_cache, all_hidden_states, all_self_attentions
    
    def load_from_pretrained(self, glm_model:ChatGLMModel):
        glm_blocks=glm_model._modules['encoder']._modules['layers']
        offset=self.split_config.head_layer_num
        for i in range(offset,self.num_layers+offset):
            self.layers[i-offset]=glm_blocks[i]   
        pass

class TailChatGLMTransformer(nn.Module):
    """Splitted Transformer class."""


    def __init__(self, glm_config: ChatGLMConfig,split_config:SplitModelConfig, device=None):
        super(TailChatGLMTransformer, self).__init__()
        self.split_config=split_config

        self.fp32_residual_connection = glm_config.fp32_residual_connection
        self.post_layer_norm = glm_config.post_layer_norm

        # Number of layers in head model.
        self.num_layers = split_config.tail_layer_num

        # Transformer layers.
        def build_layer(layer_number):
            return GLMBlock(glm_config, layer_number, device=device)

        self.layers = torch.nn.ModuleList([build_layer(i + 1) for i in range(self.num_layers)])

        #tail 部分存在layernorm
        if self.post_layer_norm:
            LayerNormFunc = RMSNorm if glm_config.rmsnorm else LayerNorm
            # Final layer norm before output.
            self.final_layernorm = LayerNormFunc(glm_config.hidden_size, eps=glm_config.layernorm_epsilon, device=device,
                                                 dtype=glm_config.torch_dtype)

        self.gradient_checkpointing = False

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def forward(
            self,             
            hidden_states, 
            attention_mask,
            rotary_pos_emb,
            presents,
            output_hidden_states,
            use_cache, 
            all_hidden_states, 
            all_self_attentions
    ):
        # if not kv_caches:
        #     kv_caches = [None for _ in range(self.num_layers)]
        # presents = () if use_cache else None
        # if self.gradient_checkpointing and self.training:
        #     if use_cache:
        #         logger.warning_once(
        #             "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
        #         )
        #         use_cache = False

        # all_self_attentions = None
        # all_hidden_states = () if output_hidden_states else None
        kv_caches = [None for _ in range(self.num_layers)]
        for index in range(self.num_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer = self._get_layer(index)
            if self.gradient_checkpointing and self.training:
                layer_ret = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_caches[index],
                    use_cache
                )
            else:
                layer_ret = layer(
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_cache=kv_caches[index],
                    use_cache=use_cache
                )
            hidden_states, kv_cache = layer_ret
            if use_cache:
                presents = presents + (kv_cache,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Final layer norm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, presents, all_hidden_states, all_self_attentions

    def load_from_pretrained(self, glm_model:ChatGLMModel):
        glm_blocks=glm_model._modules['encoder']._modules['layers']
        offset=self.split_config.head_layer_num+self.split_config.server_layer_num
        for i in range(offset,self.num_layers+offset):
            self.layers[i-offset]=glm_blocks[i]   
        #tail 部分存在layernorm
        if self.post_layer_norm:
            self.final_layernorm=glm_model._modules['encoder']._modules['final_layernorm']
        pass
    
    
#修改ChatGLMModel
class HeadChatGLMModel(ChatGLMPreTrainedModel):
    """Head model class for ChatGLM."""
    def __init__(self, glm_config: ChatGLMConfig, split_config:SplitModelConfig, device=None,empty_init=True):
        super().__init__(glm_config)
        if empty_init:
            init_method = skip_init
        else:
            init_method = default_init
        init_kwargs = {}
        if device is not None:
            init_kwargs["device"] = device
        self.embedding = init_method(Embedding, glm_config, **init_kwargs)
        self.num_layers = split_config.head_layer_num
        self.multi_query_group_num = glm_config.multi_query_group_num
        self.kv_channels = glm_config.kv_channels

        # Rotary positional embeddings
        self.seq_length = glm_config.seq_length
        rotary_dim = (
            glm_config.hidden_size // glm_config.num_attention_heads if glm_config.kv_channels is None else glm_config.kv_channels
        )
        #位置编码
        self.rotary_pos_emb = RotaryEmbedding(rotary_dim // 2, original_impl=glm_config.original_rope, device=device,
                                              dtype=glm_config.torch_dtype)
        #主要需要拆分的地方
        # self.encoder = init_method(GLMTransformer, glm_config, **init_kwargs)
        self.encoder_head = init_method(HeadChatGLMTransformer,glm_config, split_config, device=device)
        #head 部分不存在output_layer
        # self.output_layer = init_method(nn.Linear, glm_config.hidden_size, glm_config.padded_vocab_size, bias=False,
        #                                 dtype=glm_config.torch_dtype, **init_kwargs)
        self.pre_seq_len = glm_config.pre_seq_len
        self.prefix_projection = glm_config.prefix_projection
        if self.pre_seq_len is not None:
            for param in self.parameters():
                param.requires_grad = False
            self.prefix_tokens = torch.arange(self.pre_seq_len).long()
            self.prefix_encoder = PrefixEncoder(glm_config)
            self.dropout = torch.nn.Dropout(0.1)

    def get_input_embeddings(self):
        return self.embedding.word_embeddings

    def get_prompt(self, batch_size, device, dtype=torch.half):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        past_key_values = self.prefix_encoder(prefix_tokens).type(dtype)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.num_layers * 2,
            self.multi_query_group_num,
            self.kv_channels
        )
        # seq_len, b, nh, hidden_size
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 1, 0, 3, 4]).split(2)
        return past_key_values

    def forward(
            self,
            input_ids:Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.BoolTensor] = None,
            full_attention_mask: Optional[torch.BoolTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        print(f'input_ids:{input_ids.shape}')
        batch_size, seq_length = input_ids.shape

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if self.pre_seq_len is not None:
            if past_key_values is None:
                past_key_values = self.get_prompt(batch_size=batch_size, device=input_ids.device,
                                                  dtype=inputs_embeds.dtype)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask.new_ones((batch_size, self.pre_seq_len)),
                                            attention_mask], dim=-1)

        if full_attention_mask is None:
            if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)

        # Rotary positional embeddings
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

        # Run encoder_head.这里是需要拆分的地方
        hidden_states,attention_mask,rotary_pos_emb,\
            presents,output_hidden_states,use_cache, all_hidden_states, all_self_attentions = self.encoder_head(
            inputs_embeds, full_attention_mask, rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values, use_cache=use_cache, output_hidden_states=output_hidden_states
        )

        return hidden_states,attention_mask,rotary_pos_emb,presents,\
            output_hidden_states,use_cache, all_hidden_states, all_self_attentions,return_dict

    def quantize(self, weight_bit_width: int):
        from .quantization import quantize
        quantize(self.encoder_head, weight_bit_width)
        return self
    
    def load_from_pretrained(self,pretrained_model:ChatGLMForConditionalGeneration):
        glm_model=pretrained_model._modules['transformer']
        self.embedding=glm_model._modules['embedding']
        self.rotary_pos_emb=glm_model._modules['rotary_pos_emb']
        self.encoder_head.load_from_pretrained(glm_model)
        pass
    
class ServerChatGLMModel(ChatGLMPreTrainedModel):
    """Server model class for ChatGLM."""
    def __init__(self, glm_config: ChatGLMConfig,split_config:SplitModelConfig, device=None, empty_init=True):
        super().__init__(glm_config)
        if empty_init:
            init_method = skip_init
        else:
            init_method = default_init
        init_kwargs = {}
        if device is not None:
            init_kwargs["device"] = device
        self.encoder_server = init_method(ServerChatGLMTransformer, glm_config,split_config, **init_kwargs)
        # self.pre_seq_len = glm_config.pre_seq_len
        # self.prefix_projection = glm_config.prefix_projection
        # if self.pre_seq_len is not None:
        #     for param in self.parameters():
        #         param.requires_grad = False
        #     self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        #     self.prefix_encoder = PrefixEncoder(glm_config)
        #     self.dropout = torch.nn.Dropout(0.1)

    # def get_input_embeddings(self):
    #     return self.embedding.word_embeddings
    
    # server 部分用不到
    # def get_prompt(self, batch_size, device, dtype=torch.half):


    def forward(
            self,
            hidden_states_from_head,
            full_attention_mask: Optional[torch.BoolTensor] = None,
            rotary_pos_emb: Optional[torch.Tensor] = None,
            presents: Optional[Tuple[torch.Tensor, ...]] = None,
            use_cache: Optional[bool] = None,
            all_hidden_states: Optional[Tuple[torch.Tensor, ...]] = None,
            all_self_attentions: Optional[Tuple[torch.Tensor, ...]] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):

        # Run encoder.
        hidden_states,attention_mask,rotary_pos_emb,presents,\
            output_hidden_states,use_cache, all_hidden_states, all_self_attentions = self.encoder_server.forward(
            hidden_states_from_head, 
            attention_mask=full_attention_mask, 
            rotary_pos_emb=rotary_pos_emb,
            # kv_caches=None,#head server和tail不共享kv_caches
            presents=presents,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            all_hidden_states=all_hidden_states,
            all_self_attentions=all_self_attentions
        )#这里的参数全来自于head的输出
            
        return hidden_states,attention_mask,rotary_pos_emb,presents,\
            output_hidden_states,use_cache, all_hidden_states, all_self_attentions,return_dict

    def quantize(self, weight_bit_width: int):
        from .quantization import quantize
        quantize(self.encoder_server, weight_bit_width)
        return self
    
    def load_from_pretrained(self,pretrained_model:ChatGLMForConditionalGeneration):
        glm_model=pretrained_model._modules['transformer']
        self.encoder_server.load_from_pretrained(glm_model)
        pass
        
class TailChatGLMModel(ChatGLMPreTrainedModel):
    
    """Tail model class for ChatGLM."""
    def __init__(self, glm_config: ChatGLMConfig,split_config:SplitModelConfig, device=None, empty_init=True):
        super().__init__(glm_config)
        if empty_init:
            init_method = skip_init
        else:
            init_method = default_init
        init_kwargs = {}
        if device is not None:
            init_kwargs["device"] = device
        self.num_layers = split_config.tail_layer_num

        self.encoder_tail = init_method(TailChatGLMTransformer, glm_config,split_config, **init_kwargs)
        self.output_layer = init_method(nn.Linear, glm_config.hidden_size, glm_config.padded_vocab_size, bias=False,
                                        dtype=glm_config.torch_dtype, **init_kwargs)

    def get_input_embeddings(self):
        return self.embedding.word_embeddings

    #tail 部分用不到
    # def get_prompt(self, batch_size, device, dtype=torch.half):

    def forward(
            self,
            hidden_states_from_server,
            full_attention_mask: Optional[torch.BoolTensor] = None,
            rotary_pos_emb: Optional[torch.Tensor] = None,
            # kv_caches: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            presents: Optional[Tuple[torch.Tensor, ...]] = None,
            use_cache: Optional[bool] = None,
            all_hidden_states: Optional[Tuple[torch.Tensor, ...]] = None,
            all_self_attentions: Optional[Tuple[torch.Tensor, ...]] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        # Run encoder.
        hidden_states, presents, all_hidden_states, all_self_attentions=self.encoder_tail.forward(
            hidden_states_from_server, 
            attention_mask=full_attention_mask, 
            rotary_pos_emb=rotary_pos_emb,
            # kv_caches=None,#head server和tail不共享kv_caches
            presents=presents,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            all_hidden_states=all_hidden_states,
            all_self_attentions=all_self_attentions
        )#这里的参数全来自于head的输出

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def quantize(self, weight_bit_width: int):
        from .quantization import quantize
        quantize(self.encoder_tail, weight_bit_width)
        return self

    def load_from_pretrained(self,pretrained_model:ChatGLMForConditionalGeneration):
        glm_model=pretrained_model._modules['transformer']
        self.encoder_tail.load_from_pretrained(glm_model)
        self.output_layer=glm_model._modules['output_layer']
        pass

#加载预训练好的GLMTransformer，并分割成head、server、tail
def split_glm(pretained_model:ChatGLMForConditionalGeneration, split_config:SplitModelConfig, device=None)->Tuple[HeadChatGLMModel,ServerChatGLMModel,TailChatGLMModel]:
    head_model=HeadChatGLMModel(glm_config=pretained_model.config,split_config=split_config,device=device,empty_init=True)
    head_model.load_from_pretrained(pretained_model)
    server_model=None
    if split_config.with_server:
        server_model=ServerChatGLMModel(glm_config=pretained_model.config,split_config=split_config,device=device,empty_init=True)
        server_model.load_from_pretrained(pretained_model)
    tail_model=TailChatGLMModel(glm_config=pretained_model.config,split_config=split_config,device=device,empty_init=True)
    tail_model.load_from_pretrained(pretained_model)
    return head_model,server_model,tail_model
    pass