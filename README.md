<div align="center">

<h1 >DualGuard: A Parameter Space Transformation Approach for Bidirectional Defense in Split-Based LLM Fine-Tuning</h1>

</div>

This **repository** is associated with the paper "**DualGuard: A Parameter Space Transformation Approach for Bidirectional Defense in Split-Based LLM Fine-Tuning**" [<a href="https://github.com/ZJUWalker/DualGuard/edit/main/DualGuard.pdf">PDF</a>] accepted at
<a href="https://2025.aclweb.org/">The 63rd Annual Meeting of the Association for Computational Linguistics (ACL main'25)</a>

The paper introduces **DualGuard**, a novel defense mechanism designed for protecting privacy during split-based fine-tuning of large language models (LLM-FT). Split learning (SL) provides a secure way for clients to collaborate with remote servers without transmitting raw data, but it is vulnerable to data reconstruction attacks (DRAs) exploiting intermediate activations and gradients. DualGuard addresses these risks by employing a **bidirectional defense strategy**, combining local warm-up parameter space transformation and global fine-tuning parameter space retention.

**Key innovations** include:

1. A **local warm-up phase** that transforms the client-side model parameters to a secure space before formal training, making it difficult for attackers to use prior knowledge from the pre-trained model.
2. A **global fine-tuning retention strategy** that ensures the tail model does not revert to a vulnerable pre-trained state, preventing backward gradient-matching attacks.
3. **DualGuard** effectively defends against both forward and backward DRAs, demonstrating superior protection while maintaining robust downstream task performance. Detailed in the paper.

Experimental results show that DualGuard significantly outperforms existing defense methods in mitigating privacy risks, such as data reconstruction, without sacrificing the model's task accuracy.

## Quick Start

### Environment Setup

```shell
conda create -n dualguard python=3.12
conda activate dualguard
pip install -e . # setup.py
```
### Environment Config
see env file in `dualguard/utils/env.py`
```python
project_root = 'xxx'  # your project root path, code repo dir

data_root = '/share' # static data root path
model_path = '/share/models/' # model root path (gpt2-large, llama, etc.)
nltk_path = 'xxx/nltk_data/wordnet/'  # your nltk_data path (must need)

# these variables are extended from the original implementation, needn't to be changed
dataset_cache_dir = f'{data_root}/datasets/' # datasets path
model_download_dir = f'{data_root}/models/'
model_cache_dir = f'{data_root}/cache/'
attacker_path = f'{data_root}/models/attacker/'
mapper_path = f'{data_root}/models/mapper/'
reducer_path = f'{data_root}//models/reducer/'
lora_path = f'{data_root}/models/lora/'

# these variables below are used for model storage dir after fine-tuning
sip_model_dir = f'{project_root}/dualguard/static/sip'
warmup_model_dir = f'{project_root}/dualguard/static/warmup'
usl_model_dir = f'{project_root}/dualguard/static/usl'


# these variables below are used for log dir
log_dir = f'{project_root}/dualguard/log'
sip_log_dir = f'{log_dir}/sip'
usl_log_dir = f'{log_dir}/usl'
warmup_log_dir = f'{log_dir}/warmup'
eval_log_dir = f'{log_dir}/eval'
```


### Model Download
Please download the pre-trained models from <a href='https://huggingface.co/models'>huggingface hub</a> or <a href='https://www.modelscope.cn/models'>modelscope</a> and save them in the `model_path` in `dualguard/utils/env.py`. Here is the example bash script to download the pre-trained models by `modelscope`:
```shell
modelscope download --model openai-community/gpt2-large --local_dir $model_path
```
the models we used in our experiments are:
-  <a href='https://huggingface.co/openai-community/gpt2-large'>gpt2-large</a>
-  <a href='https://huggingface.co/meta-llama/Llama-3.2-1B'>llama3.2-1B</a>
-  <a href='https://huggingface.co/Qwen/Qwen2-1.5B'>qwen2-1.5B</a>
### Download Datasets
```shell
cd $dataset_cache_dir
git clone https://huggingface.co/datasets/wikitext.git
git clone https://huggingface.co/datasets/piqa.git
git clone https://huggingface.co/datasets/HuggingFaceH4/CodeAlpaca_20K.git
git clone https://huggingface.co/datasets/knkarthick/dialogsum.git
git clone https://huggingface.co/datasets/gsm8k.git
git clone https://huggingface.co/datasets/imdb.git
git clone https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese.git
git clone https://huggingface.co/datasets/frgfm/imagewoof.git
git clone https://huggingface.co/datasets/SetFit/qnli.git
git clone https://huggingface.co/datasets/linxinyuan/cola.git

```
NLTK data is used for wordnet. You can download it by:
```python
import nltk
nltk.download()
```
After you download the nltk dataset, move the dir to `nltk_path` in `dualguard/utils/env.py`.


### Run Training Experiments
Due to the time limitation, we didn't provide shell scripts for running experiments. All the experiments are run by python files, and all variables need to be set manually. <br>
Here are the steps to run experiments:

### Step1:
**Train a SIP model**, source code in `dualguard/experiment/train/sip_model_train.py`

### Step2:
**Warmup Training**, source code in `dualguard/experiment/train/warmup_train.py`

### Step3:
This is one of the most **important** step, fine-tune the **dualguard** models, source code in `dualguard/experiment/train/usl_formal_train.py` <br>
all the fine-tune method configs are defined in file `dualguard/experiment/method_config.py` <br>
It contains the following fine-tune methods:
- `lora_config`: adopt the **LORA** method to fine-tune the models
- `naive_config`: **naive** fine-tune method , needn't step 1 and 2
- `dp_forward_config`: **dp-forward** fine-tune method , needn't step 1 and 2
- `dxp_config`: **dxp** fine-tune method , needn't step 1 and 2
- `dp_sgd_config`: **dp-sgd** fine-tune method , needn't step 1 and 2
- `dualguard_config`: **dualguard** fine-tune method , must run step 1 and 2 first

####  Note that All noise-based approaches have been modified from the original implementation, due to the **semi-white box** nature of U-shaped Split Learning.

### Run Attacking Experiments
All of the attack methods are implemented in dir `dualguard/attack`, and some of them are modified from <a href='https://github.com/StupidTrees/SplitLLM'>BISR Codebase</a><br>
We declared **5** attack methods below:
```python
class AttackMethod(enum.Enum):
    SIP_ATTACK = 'sip_attack'
    TAG_ATTACK = 'tag_attack'
    LAMP_ATTACK = 'lamp_attack'
    BISR_ATTACK = 'bisr_attack'
    CONNECT_TO_PT_TAIL = 'connet_to_pt_tail'
```
U can run the attack experiments and config attack methods by python files in `dualguard/experiment/defense/attack_exp.py`.<br>Note that the attacks can only run after model fine-tuning.Here is an example.
```python
total_args = [dualguard_config]  # fine tune method config
eval_methods = [AttackMethod.CONNECT_TO_PT_TAIL]  # attack method config
```
`attack_exp.py` will run all the attack methods and fine-tuned models in loop.
