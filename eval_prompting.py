import numpy as np
import lm_eval
import torch
from transformers import AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from datamodule import GLUEDataModule
from training import OPTClassifier
import models.opt.convert_checkpoint_opt as convert_checkpoint_opt
# import models.opt.modeling_opt as modeling_opt
# import models.opt.modeling_opt_gqa as modeling_opt_gqa
# import models.llama.modeling_llama as modeling_llama
# import models.llama.modeling_llama_gqa as modeling_llama_gqa
import models.flant5.modeling_t5 as modeling_t5
import models.flant5.modeling_t5_gqa as modeling_t5_gqa
import models.flant5.modeling_t5_lora as modeling_t5_lora
import models.flant5.modeling_t5_gqa_lora as modeling_t5_gqa_lora

def evaluate(model, tasks=[], batch_size=None, max_batch_size=None):
    lm = lm_eval.api.registry.get_model("hf")(
        pretrained=model,
        batch_size=4,
        # max_batch_size=max_batch_size,
    )
    task_manager = lm_eval.tasks.TaskManager()
    results = lm_eval.simple_evaluate(model=lm, tasks=tasks, num_fewshot=0,
                                      batch_size=4, device="cuda:0",
                                      use_cache="./google/t5-small")
    print(type(results))
    print(results.keys())
    print(results["results"])
    return results

if __name__ == "__main__":
    model_name = 'google/t5-small'
    task_name = "sst2"
    num_labels = 3
    num_samples = 300
    model = modeling_t5.T5ForConditionalGeneration.from_pretrained(model_name, cache_dir="/home/qtr/.cache/huggingface/hub/").cuda()
    evaluate(model, ["mmlu"])

