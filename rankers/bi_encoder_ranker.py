import itertools
import random
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from sentence_transformers import SentenceTransformer, SimilarityFunction
from datasets import load_dataset, ClassLabel

import evaluate
import logging

logging.basicConfig(level=logging.INFO)

BATCH_SIZE = 10

class BiEncoderRanker:

    def get_scores(self, query):
        query_embedding = self.__model.encode(query, convert_to_tensor=True, show_progress_bar=False)
        
        sim_scores = self.__model.similarity(query_embedding, self.__corpus_embedding)
        
        sim_scores = sim_scores.cpu()
        return np.array(sim_scores)

    def __init__(self, model_name, corpus, similarity_function=SimilarityFunction.COSINE):
        self.__model = SentenceTransformer(model_name, similarity_fn_name=similarity_function)
        self.__corpus_embedding = self.__model.encode(corpus, convert_to_tensor=True)

        self.__logger = logging.getLogger(__name__)
        self.__logger.setLevel(logging.INFO)