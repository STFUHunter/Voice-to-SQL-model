import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

#downloading the data (for training)
rawData = load_dataset("gretelai/synthetic_text_to_sql", split="train")

#importing the pre-trained wights and initializing with it the tokenizer
model_id = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

#custom pytorch dataset wrapper class
class TextToSQL(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_len=512):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        #combining the input text with the ttarget text into the model's input
        input_text = f"Schema:\n{item['sql_context']}\n\nQuestion:\n{item['sql_prompt']}\n\nSQL Query:"
        target_text = item['sql']

        #tokenizing input and targets into tensors
        model_inputs = self.tokenizer(
            input_text,
            max_length = self.max_len,
            padding = "max_length",
            truncation = True,
            return_tensors = 'pt'
        )

        labels = self.tokenizer(
            target_text,
            max_length = self.max_len,
            padding = "max_length",
            truncation = True,
            return_tensors = 'pt'
        )

        #returning the model's batch by squeezing what was added by the tokenizer
        return{
            'input_ids' : model_inputs['input_ids'].squeeze(0),
            'attention_mask' : model_inputs['attention_mask'].squeeze(0),
            'labels' : labels['input_ids'].squeeze(0)
        }

#init the class and then sending it to the dataloader
dataset = TextToSQL(rawData, tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle= True)

for batch in dataloader:
    print("Input Tensor Shape:", batch["input_ids"].shape)
    print("Labels Tensor Shape:", batch["labels"].shape)
    print("Tensor Type Check:", type(batch["input_ids"]))
    break