import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

#Downloading the data (for training)
rawData = load_dataset("gretelai/synthetic_text_to_sql", split="train")

#Importing the pre-trained weights and initializing the tokenizer
model_id = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

#Custom PyTorch dataset wrapper class
class TextToSQL(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_len=512):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        #define prompt and target text
        prompt = f"Schema:\n{item['sql_context']}\n\nQuestion:\n{item['sql_prompt']}\n\nSQL Query:"
        target = f"{item['sql']}{self.tokenizer.eos_token}"

        #tokenize prompt and target separately to track lengths
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]

        #combine into a single full sequence
        input_ids = prompt_ids + target_ids
        
        #create labels: -100 for prompt tokens (ignored in loss), actual IDs for target tokens
        labels = [-100] * len(prompt_ids) + target_ids

        #handle truncation if sequence exceeds max_len
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            labels = labels[:self.max_len]
        else:
            #pad up to max_len
            padding_length = self.max_len - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            labels = labels + [-100] * padding_length  # Pad tokens should also be ignored (-100)

        #build attention mask (1 for real tokens, 0 for padding tokens)
        attention_mask = [1 if token != self.tokenizer.pad_token_id else 0 for token in input_ids]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

#init the class and send it to the dataloader
dataset = TextToSQL(rawData, tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch in dataloader:
    print("Input Tensor Shape:", batch["input_ids"].shape)
    print("Labels Tensor Shape:", batch["labels"].shape)
    print("Tensor Type Check:", type(batch["input_ids"]))
    break