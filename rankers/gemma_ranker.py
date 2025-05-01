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

def build_prompt(query, doc1, doc2):
    prompt =  f"You are a science researcher. Compare the documents A and B in relation with provided short text. You should output 1 with document A is more relevant and -1 otherwise. \nText: {query}\nDoc A: {doc1}\nDoc B:{doc2}.\nAnswer:"
    messages = [
        {"role": "user", "content": prompt}
    ]
    return messages

class GemmaRanker:

    def compare(self, query, doc1, doc2):
        message = build_prompt(query, doc1["abstract"], doc2["abstract"])
        output = self.__pipe(message, max_new_tokens=50)[0]["generated_text"]
        original_prompt = output[0]["content"]
        response = output[1]["content"]
        # print("Output:", response)
        final_answer = 0

        try:
            final_answer = int(response)
        except:
            try:
                lt_index = response.index('A')
            except:
                lt_index = math.inf

            try:
                gt_index = response.index('B')
            except:
                gt_index = math.inf
            
            final_answer = 1 if lt_index < gt_index else -1
        
        # print(f"Final Answer: {final_answer}")
        return final_answer

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

    def __init__(self):
        self.__pipe = pipeline("text-generation", model="google/gemma-3-4b-it", device="cuda", torch_dtype=torch.bfloat16)
        
        self.__logger = logging.getLogger(__name__)
        self.__logger.setLevel(logging.DEBUG)