from functools import partial
from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import Qwen2ForCausalLM, Qwen2Model, Qwen2Tokenizer, Qwen2Config, Qwen2PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm
from transformers.utils import (
    logging,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import get_peft_model, LoraConfig, TaskType
from functools import partial

from dualguard.defense.dp_config import *
from dualguard.defense.dp_noise import get_noise_multiplier
from dualguard.defense.dp_config import DPConfig
from dualguard.usl.split_config import Intermediate, SplitModelConfig
from dualguard.usl.split_model import SplitModel

logger = logging.get_logger(__name__)


def get_param_count(model: Qwen2PreTrainedModel):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


'''python
    qwen模型的拆分实现和包装
    QwenSplitModel是包装
    使用方法
    model_path='model path'
    tokenizer = QwenTokenizer.from_pretrained(model_path)
    qwen = QwenForCausalLM.from_pretrained(model_path)
    split_qwen=QwenSplitModel(qwen,SplitModelConfig(2,28,2,True,True))
'''


class QwenSplitModel(nn.Module):
    def __init__(
        self,
        qwen: Qwen2PreTrainedModel,
        split_config: SplitModelConfig,
        dp_config: DPConfig = DPConfig(),
        add_lora=False,
        lora_config: LoraConfig = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.config = qwen.config
        self.split_config = split_config
        self.dp_config = dp_config
        self.head_model, self.server_model, self.tail_model = split_qwen(qwen, split_config, dp_config)
        if add_lora:
            self.head_model = get_peft_model(self.head_model, lora_config)
            self.server_model = get_peft_model(self.server_model, lora_config)
            self.tail_model = get_peft_model(self.tail_model, lora_config)

    def enable_dp(self):
        self.dp_config.add_noise = True

    def disable_dp(self):
        self.dp_config.add_noise = False

    def reset_noise_multiplier(self, dataset_size: int, batch_size: int, epoch: int):
        if self.dp_config.epsilon != -1:
            self.dp_config.noise_factor = get_noise_multiplier(
                self.dp_config.epsilon,
                self.dp_config.delta,
                batch_size=batch_size,
                dataset_size=dataset_size,
                epoch=epoch,
                local_dp=self.dp_config.local_dp,
                noise_type=self.dp_config.noise_type,
            )
        else:
            self.dp_config.noise_factor = 0
            return

    def get_memory_size(self):
        param_size_on_cpu = 0
        param_size_on_gpu = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                if param.device.type == 'cpu':
                    param_size_on_cpu += param.numel() * param.dtype.itemsize
                elif param.device.type == 'cuda':
                    param_size_on_gpu += param.numel() * param.dtype.itemsize
                else:
                    print(f'unknown device type')
        return param_size_on_cpu, param_size_on_gpu

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,  # 代码目前有小bug，这里目前建议use_cache=False，否则可能影响推理
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        # return_dict: Optional[bool] = None,True
        cache_position: Optional[torch.LongTensor] = None,
        with_server: Optional[bool] = None,
        lm_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, QwenForCausalLM

        >>> model = QwenForCausalLM.from_pretrained("meta-qwen/Qwen-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-qwen/Qwen-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        temp_outputs = self.head_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
        )
        # print(f'head output: {temp_outputs[0]}')
        if with_server and self.split_config.with_server:
            temp_outputs = self.server_model(
                hidden_status_from_head=temp_outputs[0],
                attention_mask=temp_outputs[1],
                position_ids=temp_outputs[2],
                past_key_values=temp_outputs[3],
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                all_hidden_states=temp_outputs[7],
                all_self_attns=temp_outputs[8],
                return_legacy_cache=temp_outputs[9],
            )
            # print(f'server output: {temp_outputs[0]}')
        tail_outputs = self.tail_model(
            hidden_status_from_server=temp_outputs[0],
            attention_mask=temp_outputs[1],
            position_ids=temp_outputs[2],
            past_key_values=temp_outputs[3],
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            all_hidden_states=temp_outputs[7],
            all_self_attns=temp_outputs[8],
            return_legacy_cache=temp_outputs[9],
            labels=labels,
            lm_mask=lm_mask,
        )
        return tail_outputs

    def with_dp(self, switch: bool = True):
        self.head_model.with_dp(switch)
        self.server_model.with_dp(switch)
        self.tail_model.with_dp(switch)

    def get_gradiant_from_server_to_head(self):
        return self.intermediate[Intermediate.SERVER2HEAD_GRADIANT]

    def get_gradiant_from_tail_to_server(self):
        return self.intermediate[Intermediate.TAIL2SERVER_GRADIANT]

    def get_activation_from_head_to_server(self):
        return self.intermediate[Intermediate.HEAD2SERVER_ACTIVATION]

    def get_activation_from_server_to_tail(self):
        return self.intermediate[Intermediate.SERVER2TAIL_ACTIVATION]

    def _register_grad_hooks(self):
        self.intermediate = {}
        self._activation_and_grad_hooks = []

        def grad_wrapper(name: str):
            def store_grad(module, grad_input, grad_output):
                # print(f'name,{name},backward model {module.__class__.__name__},grad_input: {grad_input}')
                self.intermediate[name] = grad_input[0].detach() if len(grad_input) > 0 else ()

            return store_grad

        def activation_wrapper(name: str):
            def store_activation(module, activation_input, activation_output):
                self.intermediate[name] = activation_output[0].detach() if len(activation_output) > 0 else ()

            return store_activation

        self._activation_and_grad_hooks.append(
            self.server_model.layers[0].register_full_backward_hook(grad_wrapper(name=Intermediate.SERVER2HEAD_GRADIANT))
        )
        self._activation_and_grad_hooks.append(
            self.tail_model.layers[0].register_full_backward_hook(grad_wrapper(name=Intermediate.TAIL2SERVER_GRADIANT))
        )
        self._activation_and_grad_hooks.append(self.head_model.register_forward_hook(activation_wrapper(name=Intermediate.HEAD2SERVER_ACTIVATION)))
        self._activation_and_grad_hooks.append(self.server_model.register_forward_hook(activation_wrapper(name=Intermediate.SERVER2TAIL_ACTIVATION)))

    def destroy_grad_hooks(self):
        for hook in self._activation_and_grad_hooks:
            hook.remove()
        self._activation_and_grad_hooks = []


class QwenHead(SplitModel):

    def __init__(self, config: Qwen2Config, split_config: SplitModelConfig, dp_config: DPConfig = None):
        super().__init__(config, split_config, dp_config)
        self.embed_tokens = None
        if split_config.head_layer_num <= 0:
            logger.warning_once(
                'there is no head layer in the model, please check the split_config.head_layer_num ,\
                at least greater than 0'
            )
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.gradient_checkpointing = False
        # self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

    def _load_weight_from_pretrained_model_logically(self, pretrained_model: Qwen2ForCausalLM, from_l, to_l):
        emb_layer = pretrained_model.model.embed_tokens
        self.embed_tokens = emb_layer
        hidden_layers = pretrained_model.model.layers
        hidden_layers: List[Qwen2DecoderLayer]
        self.layers = nn.ModuleList()
        for i in range(from_l, to_l):
            hidden_layers[i].self_attn.layer_idx = i
            self.layers.append(hidden_layers[i])

    def _load_weight_from_pretrained_model_physically(self, pretrained_model: Qwen2ForCausalLM, from_l, to_l):
        self.embed_tokens = nn.Embedding(pretrained_model.config.vocab_size, pretrained_model.config.hidden_size, self.padding_idx)
        emb_layer = pretrained_model.model.embed_tokens
        self.embed_tokens.load_state_dict(emb_layer.state_dict())
        hidden_layers = pretrained_model.model.layers
        hidden_layers: List[Qwen2DecoderLayer]
        self.layers = nn.ModuleList([Qwen2DecoderLayer(self.config, layer_idx) for layer_idx in range(to_l - from_l)])
        for i in range(from_l, to_l):
            self.layers[i - from_l].load_state_dict(hidden_layers[i].state_dict())

    def load_from_pretrained_model(self, pretrained_model: Qwen2ForCausalLM, logical=True):
        from_l = 0
        to_l = self.split_config.head_layer_num
        if logical:
            self._load_weight_from_pretrained_model_logically(pretrained_model, from_l, to_l)
        else:
            self._load_weight_from_pretrained_model_physically(pretrained_model, from_l, to_l)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # --------------------------------------------------------------------------处理输入tokens
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # --------------------------------------------------------------------------处理变量
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        causal_mask = _update_causal_mask(self, attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)
        # --------------------------------------------------------------------------变量处理完毕，准备前向传播

        # 输入embedding
        hidden_states = inputs_embeds
        if DP_EMBEDDING in self.dp_config.noise_positions:  # 在激活层加入噪声
            hidden_states = self.dxp(hidden_states, embed_tokens=self.embed_tokens)

        # decoder layers
        # 初始化一些中间状态
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                # 应该用不到
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        # add hidden states from the last decoder layer
        # if output_hidden_states:
        #     all_hidden_states += (hidden_states,)
        # hidden_states = self.norm(hidden_states) #这是tailmodel需要做的事情，headmodel不需要

        # next_cache = next_decoder_cache if use_cache else None
        # if return_legacy_cache:
        #     next_cache = next_cache.to_legacy_cache()
        if DP_H2S_ACTIVATION in self.dp_config.noise_positions:  # 在激活层加入噪声
            hidden_states = self.smashed_data_dp(hidden_states, sensitivity=3.0)
        return (
            hidden_states,
            causal_mask,
            position_ids,
            past_key_values,
            output_attentions,
            use_cache,
            cache_position,
            all_hidden_states,
            all_self_attns,
            return_legacy_cache,
        )


class QwenServer(SplitModel):

    def __init__(self, config: Qwen2Config, split_config: SplitModelConfig, dp_config: DPConfig = None):
        super().__init__(config, split_config, dp_config)
        self.gradient_checkpointing = False
        if split_config.server_layer_num <= 0:
            logger.warning_once(
                'there is no server layer in the model, please check the split_config.server_layer_num ,\
                at least greater than 0'
            )

    def _load_weight_from_pretrained_model_logically(self, pretrained_model: Qwen2ForCausalLM, from_l, to_l):
        hidden_layers = pretrained_model.model.layers
        hidden_layers: List[Qwen2DecoderLayer]
        self.layers = nn.ModuleList()
        for i in range(from_l, to_l):
            hidden_layers[i].self_attn.layer_idx = i
            self.layers.append(hidden_layers[i])

    def _load_weight_from_pretrained_model_physically(self, pretrained_model: Qwen2ForCausalLM, from_l, to_l):
        hidden_layers = pretrained_model.model.layers
        hidden_layers: List[Qwen2DecoderLayer]
        self.layers = nn.ModuleList([Qwen2DecoderLayer(self.config, layer_idx + to_l) for layer_idx in range(to_l - from_l)])
        for i in range(from_l, to_l):
            self.layers[i - from_l].load_state_dict(hidden_layers[i].state_dict())

    def load_from_pretrained_model(self, pretrained_model: Qwen2ForCausalLM, logical=True):
        from_l = self.split_config.head_layer_num
        to_l = self.split_config.head_layer_num + self.split_config.server_layer_num
        if logical:
            self._load_weight_from_pretrained_model_logically(pretrained_model, from_l, to_l)
        else:
            self._load_weight_from_pretrained_model_physically(pretrained_model, from_l, to_l)

    def forward(
        self,
        hidden_status_from_head: Optional[torch.FloatTensor] = None,  # 来自head的输出隐藏状态
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        all_hidden_states: Optional[Tuple[torch.FloatTensor]] = None,
        all_self_attns: Optional[Tuple[torch.FloatTensor]] = None,
        return_legacy_cache: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # 来自head model的输出
        hidden_states = hidden_status_from_head

        # decoder layers
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        # if self.dp_config.noise_position ==DP_S2T_ACTIVATION:#在激活层加入噪声
        #     hidden_states = self.add_noise(hidden_states)
        # outputs =(hidden_states,self_attn_weights,present_key_value)

        return (
            hidden_states,
            attention_mask,
            position_ids,
            past_key_values,
            output_attentions,
            use_cache,
            cache_position,
            all_hidden_states,
            all_self_attns,
            return_legacy_cache,
        )


class QwenTail(SplitModel):
    def __init__(self, config: Qwen2Config, split_config: SplitModelConfig, dp_config: DPConfig = None):
        super().__init__(config, split_config, dp_config)
        self.gradient_checkpointing = False
        if split_config.tail_layer_num <= 0:
            logger.warning_once(
                'there is no tail layer in the model, please check the split_config.tail_layer_num ,\
                at least greater than 0'
            )

    @staticmethod
    def wrapper(module, grad_input: Tuple[torch.Tensor], grad_output: Tuple[torch.Tensor], add_noise=False, dp_func=None):
        if add_noise:
            # print(f'add noise to grad from tail to server')
            grad_input_with_noise = dp_func(grad_input[0], sensitivity=3.0, return_new_tesnor=True)
            return (grad_input_with_noise,)
        else:
            return grad_input

    def _add_nosie_on_grad_from_tail_to_server(self):
        hook_fn = partial(QwenTail.wrapper, add_noise=self.dp_config.add_noise, dp_func=self.smashed_data_dp)
        self.grad_dp_hook = self.layers[0].register_full_backward_hook(hook_fn)

    def _load_weight_from_pretrained_model_logically(self, pretrained_model: Qwen2ForCausalLM, from_l, to_l):
        hidden_layers = pretrained_model.model.layers
        hidden_layers: List[Qwen2DecoderLayer]
        self.layers = nn.ModuleList()
        for i in range(from_l, to_l):
            if not self.split_config.with_server:
                hidden_layers[i].self_attn.layer_idx = i - self.split_config.server_layer_num
            else:
                hidden_layers[i].self_attn.layer_idx = i
            self.layers.append(hidden_layers[i])
        # 加载最后的Norm和lm_head
        self.norm = pretrained_model.model.norm
        self.lm_head = pretrained_model.lm_head

    def _load_weight_from_pretrained_model_physically(self, pretrained_model: Qwen2ForCausalLM, from_l, to_l):
        hidden_layers = pretrained_model.model.layers
        hidden_layers: List[Qwen2DecoderLayer]
        self.layers = nn.ModuleList([Qwen2DecoderLayer(self.config, layer_idx + to_l) for layer_idx in range(to_l - from_l)])
        for i in range(from_l, to_l):
            self.layers[i - from_l].load_state_dict(hidden_layers[i].state_dict())
        self.norm = Qwen2RMSNorm(pretrained_model.config.hidden_size, eps=pretrained_model.config.rms_norm_eps)
        self.norm.load_state_dict(pretrained_model.model.norm.state_dict())
        self.lm_head = nn.Linear(pretrained_model.config.hidden_size, pretrained_model.config.vocab_size, bias=False)
        self.lm_head.load_state_dict(pretrained_model.lm_head.state_dict())

    def load_from_pretrained_model(self, pretrained_model: Qwen2ForCausalLM, logical=True):
        from_l = self.split_config.total_hidden_layers - self.split_config.tail_layer_num
        to_l = self.split_config.total_hidden_layers
        if logical:
            self._load_weight_from_pretrained_model_logically(pretrained_model, from_l, to_l)
        else:
            self._load_weight_from_pretrained_model_physically(pretrained_model, from_l, to_l)
        if DP_T2S_GRADIENT in self.dp_config.noise_positions:
            self._add_nosie_on_grad_from_tail_to_server()

    def forward(
        self,
        hidden_status_from_server: Optional[torch.FloatTensor] = None,  # 来自head的输出隐藏状态
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        all_hidden_states: Optional[Tuple[torch.FloatTensor]] = None,
        all_self_attns: Optional[Tuple[torch.FloatTensor]] = None,
        return_legacy_cache: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        lm_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # print(f'hidden_status_from_server shape is {hidden_status_from_server.shape}')
        # 来自server model的输出
        hidden_states = hidden_status_from_server
        # decoder layers
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        # print(f'hidden_states shape is {hidden_states.shape}')
        hidden_states = self.norm(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache and next_cache is not None:
            next_cache = next_cache.to_legacy_cache()

        # hidden_states = tail_outputs[0]
        # if self.config.pretraining_tp > 1:
        #     lm_head_slices = self.tail_model.lm_head.weight.split(self.config.vocab_size // self.config.pretraining_tp, dim=0)
        #     logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
        #     logits = torch.cat(logits, dim=-1)
        # else:
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            if lm_mask is not None:
                shift_lm_mask = lm_mask[..., 1:].contiguous()
                shift_lm_mask = shift_lm_mask.view(-1)
                loss_fct = CrossEntropyLoss(reduction='none')
                loss = loss_fct(shift_logits, shift_labels)
                loss = loss * shift_lm_mask.float()
                loss = loss.sum() / shift_lm_mask.sum()
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


def _update_causal_mask(
    partitioned_model: Qwen2PreTrainedModel,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: Cache,
    output_attentions: bool,
):
    # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
    # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
    # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
    # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

    if partitioned_model.config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None

    # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
    # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
    # to infer the attention mask.
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    using_static_cache = isinstance(past_key_values, StaticCache)

    # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
    if partitioned_model.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
        if AttentionMaskConverter._ignore_causal_mask_sdpa(
            attention_mask,
            inputs_embeds=input_tensor,
            past_key_values_length=past_seen_tokens,
            is_training=partitioned_model.training,
        ):
            return None

    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    if using_static_cache:
        target_length = past_key_values.get_max_length()
    else:
        target_length = attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else past_seen_tokens + sequence_length + 1

    if attention_mask is not None and attention_mask.dim() == 4:
        # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
        if attention_mask.max() != 0:
            raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)
    if (
        partitioned_model.config._attn_implementation == "sdpa"
        and attention_mask is not None
        and attention_mask.device.type == "cuda"
        and not output_attentions
    ):
        # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
        # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        # Details: https://github.com/pytorch/pytorch/issues/110213
        causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

    return causal_mask


# 将qwen模型分割成三个部分，head，server，tail
# 首先从与训练模型中加载模型
def split_qwen(
    pretrained_model: Qwen2ForCausalLM, split_config: SplitModelConfig, dp_config: DPConfig
) -> Tuple[QwenHead, Optional[QwenServer], QwenTail]:
    # self.split_config = split_config
    config = pretrained_model.config
    # config:QwenConfig
    # assert config.num_hidden_layers == split_config.total_hidden_layers, \
    #     'total splited layer num must equal to config.num_hidden_layers'
    if split_config.server_layer_num <= 0:
        split_config.server_layer_num = config.num_hidden_layers - split_config.tail_layer_num - split_config.head_layer_num
    head_model = QwenHead(config, split_config, dp_config)
    head_model.load_from_pretrained_model(pretrained_model, logical=split_config.logicl_load)
    print(f'head model loaded scucessfully')
    server_model = None
    if split_config.with_server:
        server_model = QwenServer(config, split_config, dp_config)
        server_model.load_from_pretrained_model(pretrained_model, logical=split_config.logicl_load)
    print(f'server model loaded scucessfully')
    tail_model = QwenTail(config, split_config, dp_config)
    tail_model.load_from_pretrained_model(pretrained_model, logical=split_config.logicl_load)
    print(f'tail model loaded scucessfully')

    return head_model, server_model, tail_model
