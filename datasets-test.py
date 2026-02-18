from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

access_token = ""

dataset = load_dataset("nyu-mll/glue", "mrpc", split="train")

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def encode(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length")

dataset = dataset.map(encode, batched=True)
print(dataset[0])

dataset = dataset.map(lambda examples: {"labels": examples["label"]}, batched=True)

dataset = dataset.select_columns(["input_ids", "token_type_ids", "attention_mask", "labels"])
dataset = dataset.with_format(type="torch")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)