import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['HTTP_PROXY'] = '172.20.110.220:7890'
os.environ['HTTPS_PROXY'] = '172.20.110.220:7890'

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from datamodule import GLUEDataModule
from training import Classifier
from models.flant5.configuration_t5_lora import T5LoraConfig
import models.flant5.modeling_t5 as modeling_t5
import models.flant5.modeling_t5_lora as modeling_t5_lora
from lora.lora_utils import (
    print_trainable_parameters,
    mark_only_lora_as_trainable
)

import toml

# tensorboard --logdir lightning_logs/ --port 6006

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_type = 't5-small'
model_name = 'google-t5/t5-small'

# tasks = [['sst2', 2], ['qnli', 2], ['mnli', 3]]
task_name = 'sst2' 
num_labels = 2
config_files = ["./lora/lora_by_type.toml"]

for config_file in config_files:
    # load toml config file
    with open(config_file, "r") as f:
        lora_config = toml.load(f)
    print(f"LoRA PEFT with {config_file} config file successfully loaded!")

peft_config = T5LoraConfig.from_pretrained(model_name, lora_config=lora_config, num_labels=num_labels)
model = modeling_t5_lora.T5ForSequenceClassification.from_pretrained(model_name, config=peft_config)
print(model)
model = mark_only_lora_as_trainable(model)

data_module = GLUEDataModule(model_name=model_name, task_name=task_name)
data_module.setup("fit")

classifier = Classifier(model)

logger = TensorBoardLogger("/home/qtr/Gqa_search/models/tensorboard/lightning_logs", name=task_name + "-" + model_type + "Lora")
trainer = pl.Trainer(max_epochs=30, logger=logger)
trainer.fit(classifier, data_module)