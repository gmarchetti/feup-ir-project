import logging
import pandas as pd
import os
from rankers.qwen_ranker import QwenRanker
from functools import cmp_to_key

PATH_COLLECTION_DATA = 'data/subtask_4b/subtask4b_collection_data.pkl'
PATH_QUERY_DATA = 'data/dev-tweets/subtask4b_query_tweets_dev.tsv'

df_query = pd.read_csv(PATH_QUERY_DATA, sep = '\t')
df_collection = pd.read_pickle(PATH_COLLECTION_DATA)

pre_sampled_docids = ['hg3xpej0', 'styavbvi', '3qvh482o', 'rthsl7a9', '1adt71pk']

sample_collection = df_collection[df_collection['cord_uid'].isin(pre_sampled_docids)]
sample_query = df_query[:5]

logging.basicConfig(level=logging.DEBUG)

def test_can_instantiate():
    llm_ranker = QwenRanker()
    assert llm_ranker != None

def test_compare():    
    llm_ranker = QwenRanker()
    score = llm_ranker.compare(sample_query.loc[0, "tweet_text"], sample_collection.iloc[0], sample_collection.iloc[1])
    assert (score == -1) or (score == 1)

def test_sort_docs():
    logging.debug(pre_sampled_docids)
    
    llm_ranker = QwenRanker()
    sorted_uids = llm_ranker.sort_docs(sample_query.loc[1, "tweet_text"], sample_collection)
    
    logging.debug(sorted_uids)
    assert len(pre_sampled_docids) == len(sorted_uids)

def test_bubble_sort_docs():
    logging.debug(pre_sampled_docids)
    
    llm_ranker = QwenRanker()
    sorted_uids = llm_ranker.sort_cached_bubble(sample_query.loc[0, "tweet_text"], sample_collection)
    
    logging.debug(sorted_uids)
    assert len(pre_sampled_docids) == len(sorted_uids)