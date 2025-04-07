import numpy as np
import os

from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, ClassLabel

import evaluate

metric = evaluate.load("accuracy")
model = AutoModelForSequenceClassification.from_pretrained("allenai/scibert_scivocab_cased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # convert the logits to their predicted class
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def preprocess_function(examples):
    full_str = "[CLS]"
    full_str += examples["query"]
    
    for key in ["authors", "title", "abstract", "journal", "time"]:
        if examples[key] != None:
            full_str += "[SEP] " + examples[key]
    
    return tokenizer(full_str, padding="max_length", max_length=512, truncation=True)

data_path = f"./data/"

train_file = os.path.join(data_path, 'train', r'tweets-query-pairs.tsv')
valid_file = os.path.join(data_path, 'valid', r'tweets-query-pairs.tsv')

train_dataset = load_dataset(data_path, split="train")
test_dataset = load_dataset(data_path, split="test")

train_dataset = train_dataset.map(preprocess_function)
test_dataset = test_dataset.map(preprocess_function)

id2label = {0: "0", 1: "1"}
label2id = {"0": 0, "1": 1}

training_args = TrainingArguments(
    output_dir="models/cross-embedding",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    fp16=True,
    optim="adamw_torch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train(resume_from_checkpoint=False)