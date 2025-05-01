import itertools
import random
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, ClassLabel

import evaluate
import logging

logging.basicConfig(level=logging.INFO)

BATCH_SIZE = 10

class CrossRanker:

    def preprocess_function(self, query, examples):
        full_str = "[CLS]"
        full_str += query
        
        for key in ["authors", "title", "abstract", "journal"]:
            if examples[key] != None:
                full_str += "[SEP] " + str(examples[key])
        
        return full_str

    def get_scores(self, query, corpus):
        query_list = []
        scores = []

        for idx, row in corpus.iterrows():
            query_list.append(self.preprocess_function(query, row))

        for batch in itertools.batched(query_list, BATCH_SIZE):
            tokenized_list = self.__tokenizer(batch, padding="max_length", max_length=512, truncation=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = self.__model(**tokenized_list)
            
            for logit in outputs.logits:
                logit_value = logit[1].item()
                self.__logger.debug(logit_value)
                scores.append(logit_value)
        
        return np.array(scores)

    def __init__(self, model_name):
        self.__model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to("cuda")
        self.__tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")

        self.__logger = logging.getLogger(__name__)
        self.__logger.setLevel(logging.INFO)