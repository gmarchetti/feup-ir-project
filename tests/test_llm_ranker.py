import logging
import pandas as pd
import os
from rankers.llm_ranker import LlmRanker
from functools import cmp_to_key
PATH_COLLECTION_DATA = 'data/subtask_4b/subtask4b_collection_data.pkl'
PATH_QUERY_DATA = 'data/dev-tweets/subtask4b_query_tweets_dev.tsv'

df_query = pd.read_csv(PATH_QUERY_DATA, sep = '\t')
df_collection = pd.read_pickle(PATH_COLLECTION_DATA)

sample_collection = df_collection[:5]
sample_query = df_query[:1]

logging.basicConfig(level=logging.DEBUG)

def test_can_instantiate():
    llm_ranker = LlmRanker()
    assert llm_ranker != None

def test_compare():    
    llm_ranker = LlmRanker()
    score = llm_ranker.compare(sample_query.loc[0, "tweet_text"], sample_collection.iloc[0], sample_collection.iloc[1])
    assert (score == -1) or (score == 1)

def test_sort_docs():
    collection_as_dict = sample_collection.to_dict('index')
    unsorted_uids = [collection_as_dict[index]["cord_uid"] for index in collection_as_dict.keys()]
    logging.debug(unsorted_uids)
    
    llm_ranker = LlmRanker()
    sorted_uids = llm_ranker.sort_docs(sample_query.loc[0, "tweet_text"], sample_collection)
    
    logging.debug(sorted_uids)
    assert len(unsorted_uids) == len(sorted_uids)