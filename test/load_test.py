import os
import sys

import torch
sys.path.append(os.path.abspath('/home/wyz/deeplearning/workspace/Privacy-USL-LLM'))


from dualguard.attack.sip import GRUDRInverter,load_sip_model

sip_config={
    'n_embed':1280,
    'vocab_size':50257,
    'hidden_size':256,
    'bidirectional':False,
}

if __name__ == '__main__':
    model_path='/home/wyz/deeplearning/workspace/Privacy-USL-LLM/dualguard/static/sip/sip_qwen2-1.5b.pth'
    # sip_model=load_sip_model(model_path,GRUDRInverter,deivce_mapping='cuda:1',**sip_config)
    # print(sip_model)
    model=torch.load(model_path,map_location='cuda:1',weights_only=False)
    # print(model)
    torch.save(model.state_dict(),model_path)