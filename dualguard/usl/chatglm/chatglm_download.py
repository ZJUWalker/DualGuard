from transformers import LlamaForCausalLM,AutoTokenizer
path='/home/wyz/deeplearning/workspace/USLattack/data/models/meta-llama/Llama-3.2-1B'
# model_dir = snapshot_download("LLM-Research/Llama-3.2-1B")
tokenizer =AutoTokenizer.from_pretrained(path, trust_remote_code=True,use_safetensors=True)
model = LlamaForCausalLM.from_pretrained(path, trust_remote_code=True)
print(model)
