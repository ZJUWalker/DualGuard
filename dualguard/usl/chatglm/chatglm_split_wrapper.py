import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import List, Tuple,Dict,Optional,Union

from transformers.modeling_outputs import CausalLMOutputWithPast

from usl.split_config import SplitModelConfig
from usl.chatglm.chatglm_split import split_glm
from usl.chatglm.chatglm_model import ChatGLMForConditionalGeneration

class SplittedChatGLM(object):
   
    def __init__(self,model_path,split_config:SplitModelConfig,*args, **kwargs):
       self.split_config=split_config
       self.original_model = ChatGLMForConditionalGeneration.from_pretrained(model_path,*args, **kwargs)
       self.config = self.original_model.config
       self.head_model,self.server_model,self.tail_model = split_glm(self.original_model,split_config)
       print(f'config: {self.config}')
       
       
    @classmethod
    def from_pretrained(cls,model_path:str,split_config:SplitModelConfig,*args, **kwargs)-> "SplittedChatGLM":
        return SplittedChatGLM(model_path,split_config,*args, **kwargs)
    
    def to(self,device):
        self.head_model.to(device)
        if self.split_config.with_server:
            self.server_model.to(device)
        self.tail_model.to(device)
    
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            return_last_logit: Optional[bool] = False,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        temp_outputs = self.head_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if self.split_config.with_server:
            temp_outputs = self.server_model(
                hidden_states_from_head=temp_outputs[0],
                full_attention_mask=attention_mask,
                rotary_pos_emb=temp_outputs[2],
                presents=temp_outputs[3],
                use_cache=use_cache,
                all_hidden_states=temp_outputs[5],
                all_self_attentions=temp_outputs[6],
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        transformer_outputs=self.tail_model.forward(
                hidden_states_from_server=temp_outputs[0],
                full_attention_mask=attention_mask,
                rotary_pos_emb=temp_outputs[2],
                presents=temp_outputs[3],
                use_cache=use_cache,
                all_hidden_states=temp_outputs[5],
                all_self_attentions=temp_outputs[6],
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
        #TODO 检查代码

        hidden_states = transformer_outputs[0]
        if return_last_logit:
            hidden_states = hidden_states[-1:]
        lm_logits = self.tail_model.output_layer(hidden_states)
        lm_logits = lm_logits.transpose(0, 1).contiguous()

        loss = None
        if labels is not None:
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    
    