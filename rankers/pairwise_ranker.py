import itertools
import math
from pydoc import doc
import random
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoModelForCausalLM
from datasets import load_dataset, ClassLabel
from functools import cmp_to_key

import logging

logging.basicConfig(level=logging.DEBUG)

class PairwiseRanker:

    def preprocess_function(self, query, abs1, abs2):
        full_str = f"[CLS] {query} [SEP] {abs1} [SEP] {abs2}"
            
        return self.__tokenizer(full_str, padding="max_length", truncation=True, return_tensors='pt')

    def compare(self, query, doc1, doc2):
        message = self.preprocess_function(query, doc1["abstract"], doc2["abstract"]).to('cuda')

        with torch.no_grad():
            logits = self.__model(**message).logits
            predicted_class_id = logits.argmax().item()

        print(f"Predicted class id: {predicted_class_id}")
        return predicted_class_id - 1

    def sort_cached_bubble(self, query, docs):
        collection_as_dict = docs.to_dict('index')
        doc_index_list = list(collection_as_dict.keys())
        cached_response = {idx: {} for idx in doc_index_list}
        idx = 0
        while idx < len(doc_index_list)-1:
            doc1_idx = doc_index_list[idx]
            doc2_idx = doc_index_list[idx+1]
            # print(f"comparing idx {collection_as_dict[doc1_idx]["cord_uid"]} with {collection_as_dict[doc2_idx]["cord_uid"]}")
            comparison_result = 0
            if doc2_idx in cached_response[doc1_idx].keys():
                # print("Using cached result")
                comparison_result = cached_response[doc1_idx][doc2_idx]
            else:
                # print("Computing comparison")
                comparison_result = self.compare(query, collection_as_dict[doc1_idx], collection_as_dict[doc2_idx]) 
                cached_response[doc1_idx][doc2_idx] = comparison_result
                cached_response[doc2_idx][doc1_idx] = -comparison_result
            
            if comparison_result < 0:
                # print("Switch places")
                temp = doc_index_list[idx+1]
                doc_index_list[idx+1] = doc_index_list[idx]
                doc_index_list[idx] = temp
                idx = -1
            
            idx += 1

        sorted_uids = [collection_as_dict[index]["cord_uid"] for index in doc_index_list]
        return sorted_uids

    def __init__(self, model_name):
        self.__model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, reference_compile=False).to("cuda")
        self.__tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

        self.__logger = logging.getLogger(__name__)
        self.__logger.setLevel(logging.INFO)