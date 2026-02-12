import os
from dataclasses import dataclass
from transformers import AutoConfig

@dataclass
class Config:
    model_path: str

    def __post_init__(self):
        assert os.path.isdir(self.model_path)
        self.hf_config = AutoConfig.from_pretrained(self.model_path)
        # print(self.hf_config)