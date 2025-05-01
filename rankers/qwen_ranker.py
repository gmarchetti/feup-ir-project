import itertools
import math
from pydoc import doc
import random
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoModelForCausalLM
from datasets import load_dataset, ClassLabel
from functools import cmp_to_key

import evaluate
import logging

logging.basicConfig(level=logging.DEBUG)

def build_prompt(query, doc1, doc2):
    prompt =  f"You are a science researcher. Compare the documents A and B in relation with provided short text. You should output 1 with document A is more relevant and -1 otherwise. \nText: {query}\nDoc A: {doc1}\nDoc B:{doc2}.\nAnswer:"
    messages = [
        {"role": "user", "content": prompt}
    ]
    return messages

class QwenRanker:

    def compare(self, query, doc1, doc2):
        message = build_prompt(query, doc1["abstract"], doc2["abstract"])
        text = self.__tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self.__tokenizer(text, return_tensors="pt").to(self.__model.device)

        # conduct text completion
        generated_ids = self.__model.generate(
            **model_inputs,
            max_new_tokens=9000
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.__tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.__tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        print("thinking content:", thinking_content)
        print("content:", content)

        try:
            lt_index = content.index('-1')
        except:
            lt_index = math.inf

        try:
            gt_index = content.index('1')
        except:
            gt_index = math.inf
        
        final_answer = 1 if lt_index < gt_index else -1
        # print(final_answer)
        return final_answer

    def sort_docs(self, query, docs):
        collection_as_dict = docs.to_dict('index')
        sorted_docs_indexes = sorted(collection_as_dict, key=cmp_to_key(lambda item1, item2: self.compare(query, collection_as_dict[item1], collection_as_dict[item2])), reverse=True)
        sorted_uids = [collection_as_dict[index]["cord_uid"] for index in sorted_docs_indexes]
        return sorted_uids

    def sort_cached_bubble(self, query, docs):
        collection_as_dict = docs.to_dict('index')
        doc_index_list = list(collection_as_dict.keys())
        cached_response = {idx: {} for idx in doc_index_list}
        idx = 0
        while idx < len(doc_index_list)-1:
            doc1_idx = doc_index_list[idx]
            doc2_idx = doc_index_list[idx+1]
            print(f"comparing idx {collection_as_dict[doc1_idx]["cord_uid"]} with {collection_as_dict[doc2_idx]["cord_uid"]}")
            comparison_result = 0
            if doc2_idx in cached_response[doc1_idx].keys():
                print("Using cached result")
                comparison_result = cached_response[doc1_idx][doc2_idx]
            else:
                print("Computing comparison")
                comparison_result = self.compare(query, collection_as_dict[doc1_idx], collection_as_dict[doc2_idx]) 
                cached_response[doc1_idx][doc2_idx] = comparison_result
                cached_response[doc2_idx][doc1_idx] = -comparison_result
            
            if comparison_result < 0:
                print("Switch places")
                temp = doc_index_list[idx+1]
                doc_index_list[idx+1] = doc_index_list[idx]
                doc_index_list[idx] = temp
                idx = -1
            
            idx += 1

        sorted_uids = [collection_as_dict[index]["cord_uid"] for index in doc_index_list]
        return sorted_uids

    def __init__(self, model_id="Qwen/Qwen3-1.7B"):
        self.__model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="cuda")
        self.__tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.__logger = logging.getLogger(__name__)
        self.__logger.setLevel(logging.DEBUG)