

<div align="center">

<h1 >DualGuard: Un Enfoque de Transformación del Espacio de Parámetros para la Defensa Bidireccional en el Ajuste Fino de LLM Basado en División</h1>

</div>

Este **repositorio** está asociado con el artículo "**DualGuard: Un Enfoque de Transformación del Espacio de Parámetros para la Defensa Bidireccional en el Ajuste Fino de LLM Basado en División**" [<a href="https://github.com/ZJUWalker/DualGuard/blob/main/DualGuard.pdf">PDF</a>] aceptado en
<a href="https://2025.aclweb.org/">La 63ª Reunión Anual de la Asociación de Lingüística Computacional (ACL principal'25)</a>

El artículo presenta **DualGuard**, un mecanismo de defensa novel diseñado para proteger la privacidad durante el ajuste fino basado en división de grandes modelos de lenguaje (LLM-FT). El aprendizaje dividido (SL) proporciona una forma segura para que los clientes colaboren con servidores remotos sin transmitir datos en bruto, pero es vulnerable a ataques de reconstrucción de datos (DRAs) que explotan activaciones y gradientes intermedios. DualGuard aborda estos riesgos mediante una **estrategia de defensa bidireccional**, que combina la transformación del espacio de parámetros de calentamiento local y la retención del espacio de parámetros en el ajuste fino global.

**Las innovaciones clave** incluyen:

1. Una **fase de calentamiento local** que transforma los parámetros del modelo del lado del cliente a un espacio seguro antes del entrenamiento formal, dificultando que los atacantes utilicen el conocimiento previo del modelo preentrenado.
2. Una **estrategia de retención en el ajuste fino global** que asegura que el modelo final no retroceda a un estado vulnerable preentrenado, previniendo ataques de emparejamiento de gradientes hacia atrás.
3. **DualGuard** se defiende eficazmente tanto contra DRAs hacia adelante como hacia atrás, demostrando una protección superior mientras mantiene un rendimiento robusto en las tareas descendentes. Detallado en el artículo.

Los resultados experimentales muestran que DualGuard supera significativamente a los métodos de defensa existentes en la mitigación de riesgos de privacidad, como la reconstrucción de datos, sin sacrificar la precisión de la tarea del modelo.

## Inicio Rápido

### Configuración del Entorno

```shell
conda create -n dualguard python=3.12
conda activate dualguard
pip install -e . # setup.py
```
### Configuración del Entorno
consulte el archivo env en `dualguard/utils/env.py`
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


### Descarga de Modelos
Descargue los modelos preentrenados desde <a href='https://huggingface.co/models'>huggingface hub</a> o <a href='https://www.modelscope.cn/models'>modelscope</a> y guárdelos en `model_path` en `dualguard/utils/env.py`. A continuación se muestra el script bash de ejemplo para descargar los modelos preentrenados mediante `modelscope`:
```shell
modelscope download --model openai-community/gpt2-large --local_dir $model_path
```
los modelos que utilizamos en nuestros experimentos son:
-  <a href='https://huggingface.co/openai-community/gpt2-large'>gpt2-large</a>
-  <a href='https://huggingface.co/meta-llama/Llama-3.2-1B'>llama3.2-1B</a>
-  <a href='https://huggingface.co/Qwen/Qwen2-1.5B'>qwen2-1.5B</a>
### Descarga de Conjuntos de Datos
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
Los datos de NLTK se utilizan para wordnet. Puedes descargarlo mediante:
```python
import nltk
nltk.download()
```
Una vez descargado el conjunto de datos de nltk, mueva el directorio a `nltk_path` en `dualguard/utils/env.py`.


### Ejecutar Experimentos de Entrenamiento
Debido a limitaciones de tiempo, no proporcionamos scripts shell para ejecutar experimentos. Todos los experimentos se ejecutan mediante archivos Python, y todas las variables deben configurarse manualmente. <br>
A continuación se detallan los pasos para ejecutar los experimentos:

### Paso 1:
**Entrenar un modelo SIP**, código fuente en `dualguard/experiment/train/sip_model_train.py`

### Paso 2:
**Entrenamiento de calentamiento (Warmup)**, código fuente en `dualguard/experiment/train/warmup_train.py`

### Paso 3:
Este es uno de los pasos más **importantes**: ajustar finamente los modelos **dualguard**, código fuente en `dualguard/experiment/train/usl_formal_train.py` <br>
todas las configuraciones de los métodos de ajuste fino están definidas en el archivo `dualguard/experiment/method_config.py` <br>
Contiene los siguientes métodos de ajuste fino:
- `lora_config`: adopta el método **LORA** para ajustar finamente los modelos
- `naive_config`: método de ajuste fino **naive**, no requiere los pasos 1 y 2
- `dp_forward_config`: método de ajuste fino **dp-forward**, no requiere los pasos 1 y 2
- `dxp_config`: método de ajuste fino **dxp**, no requiere los pasos 1 y 2
- `dp_sgd_config`: método de ajuste fino **dp-sgd**, no requiere los pasos 1 y 2
- `dualguard_config`: método de ajuste fino **dualguard**, debe ejecutar primero los pasos 1 y 2

####  Tenga en cuenta que todos los enfoques basados en ruido han sido modificados con respecto a la implementación original, debido a la naturaleza **semi-white box** del Aprendizaje Divido en Forma de U.

### Ejecutar Experimentos de Ataque
Todos los métodos de ataque están implementados en el directorio `dualguard/attack`, y algunos de ellos están modificados a partir de <a href='https://github.com/StupidTrees/SplitLLM'>BISR Codebase</a><br>
Declararemos a continuación **5** métodos de ataque:
```python
class AttackMethod(enum.Enum):
    SIP_ATTACK = 'sip_attack'
    TAG_ATTACK = 'tag_attack'
    LAMP_ATTACK = 'lamp_attack'
    BISR_ATTACK = 'bisr_attack'
    CONNECT_TO_PT_TAIL = 'connet_to_pt_tail'
```
Puede ejecutar los experimentos de ataque y configurar los métodos de ataque mediante los archivos Python en `dualguard/experiment/defense/attack_exp.py`.<br>Tenga en cuenta que los ataques solo pueden ejecutarse después del ajuste fino del modelo. A continuación se muestra un ejemplo.
```python
total_args = [dualguard_config]  # fine tune method config
eval_methods = [AttackMethod.CONNECT_TO_PT_TAIL]  # attack method config
```
 `attack_exp.py` ejecutará en un bucle todos los métodos de ataque y los modelos ajustados finamente.
