project_root = '/home/wyz/deeplearning/workspace/DualGuard'

data_root = '/share'
model_path= '/share/models/'
nltk_path='/home/wyz/nltk_data/wordnet/'
dataset_cache_dir = f'{data_root}/datasets/'
model_download_dir = f'{data_root}/models/'
model_cache_dir = f'{data_root}/cache/'
attacker_path = f'{data_root}/models/attacker/'
mapper_path = f'{data_root}/models/mapper/'
reducer_path = f'{data_root}//models/reducer/'
lora_path = f'{data_root}/models/lora/'

#用于模型和log文件的保存根路径，在使用时还需要根据模型和数据集加子目录
sip_model_dir=f'{project_root}/dualguard/static/sip'
warmup_model_dir=f'{project_root}/dualguard/static/warmup'
usl_model_dir=f'{project_root}/dualguard/static/usl'
log_dir=f'{project_root}/dualguard/log'

sip_log_dir=f'{log_dir}/sip'
usl_log_dir=f'{log_dir}/usl'
warmup_log_dir=f'{log_dir}/warmup'
eval_log_dir=f'{log_dir}/eval'