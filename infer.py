import os    

from config import Config

path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
config = Config(model_path=path)

print(type(config.hf_config)) # <class 'transformers.models.qwen3.configuration_qwen3.Qwen3Config'>
