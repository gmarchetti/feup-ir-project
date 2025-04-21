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

logging.basicConfig(level=logging.DEBUG)

class CrossRanker:

    def preprocess_function(self, query, examples):
        full_str = "[CLS]"
        full_str += query
        
        # print(examples)
        
        for key in ["authors", "title", "abstract", "journal"]:
            if examples[key] != None:
                full_str += "[SEP] " + str(examples[key])
        
        return self.__tokenizer(full_str, padding="max_length", max_length=512, truncation=True, return_tensors="pt").to("cuda")

    def score_doc_for_query(self, query, doc):
        with torch.no_grad():
            tokenized_query = self.preprocess_function(query, doc)
            logits = self.__model(**tokenized_query).logits[0][1].item()
        
        return logits

    def get_scores(self, query, corpus):

        scores = corpus.apply(lambda x: self.score_doc_for_query(query, x), axis=1)

        self.__logger.debug(scores.head())

        return scores

    def __init__(self):
        self.__model = AutoModelForSequenceClassification.from_pretrained("models/cross-embedding/checkpoint-2138", num_labels=2).to("cuda")
        self.__tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")

        self.__logger = logging.getLogger("CrossRanker")
        self.__logger.setLevel(logging.DEBUG)