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

    def preprocess_function(self, query, doc1, doc2):
        paper_data = "{authors} [SEP] {title} [SEP] {journal} [SEP] {abstract}"

        doc1_data = paper_data.format(authors=doc1["authors"],
                                           title=doc1["title"],
                                           journal=doc1["journal"],
                                           abstract=doc1["abstract"],
                                           )
        doc2_data = paper_data.format(authors=doc2["authors"],
                                           title=doc2["title"],
                                           journal=doc2["journal"],
                                           abstract=doc2["abstract"],
                                           )

        full_str = f"[CLS] {query} [SEP] {doc1_data} [SEP] {doc2_data}"
            
        return self.__tokenizer(full_str, padding="max_length", truncation=True, return_tensors='pt')

    def get_higher_rel_prob(self, query, doc1, doc2, cache={}):
        doc1_uid = doc1["cord_uid"]
        doc2_uid = doc2["cord_uid"]

        if doc1_uid in (cache.keys()):
            if doc2_uid in cache[doc1_uid].keys():
                return cache[doc1_uid][doc2_uid]

        message = self.preprocess_function(query, doc1, doc2).to('cuda')

        with torch.no_grad():
            logits = self.__model(**message).logits[0]
        
        doc1gt2 = logits[0].item()
        doc2gt1 = logits[1].item()

        # print(doc1gt2, doc2gt1)

        doc1_scores = cache.get(doc1_uid, {})
        doc1_scores[doc2_uid] = doc1gt2
        cache[doc1_uid] = doc1_scores

        doc2_scores = cache.get(doc2_uid, {})
        doc2_scores[doc1_uid] = doc2gt1
        cache[doc2_uid] = doc2_scores

        # print(f"Predicted class id: {predicted_class_id}")
        return doc1gt2


    def rank_avg_prob(self, query, docs, use_cache=True):
        # print(docs)
        collection_as_dict = docs.to_dict('index')
        doc_index_list = list(collection_as_dict.keys())
        scores = []
        cord_uids = []

        if use_cache:
            cached_result = {}

        for idx in doc_index_list:
            cord_uid = collection_as_dict[idx]["cord_uid"]
            doc1 = collection_as_dict[idx]
            other_docs = docs[docs["cord_uid"] != cord_uid]
            probs = other_docs.apply(lambda doc2: self.get_higher_rel_prob(query, doc1, doc2, cached_result), axis = 1)

            scores.append(probs.mean())
            cord_uids.append(cord_uid)
            # print(f"{cord_uids[-1]} score: {scores[-1]}")


        sorted_uid = pd.DataFrame({
            "cord_uids" : cord_uids,
            "scores" : scores
        }).sort_values(by=["scores"], ascending=False)

        return sorted_uid["cord_uids"].tolist()

    def __init__(self, model_name):
        self.__model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, reference_compile=False).to("cuda")
        self.__tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

        self.__logger = logging.getLogger(__name__)
        self.__logger.setLevel(logging.INFO)