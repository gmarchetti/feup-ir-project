import logging
import pandas as pd
import os
from rankers.cross_embedding_ranker import CrossRanker

PATH_COLLECTION_DATA = '/home/guilherme/workspace/feup-ir-project/data/subtask_4b/subtask4b_collection_data.pkl'
PATH_QUERY_DATA = '/home/guilherme/workspace/feup-ir-project/data/subtask_4b/subtask4b_query_tweets.tsv'

df_query = pd.read_csv(PATH_QUERY_DATA, sep = '\t')
df_collection = pd.read_pickle(PATH_COLLECTION_DATA)


sample_collection = df_collection[:20]
sample_query = df_query[:5]

logging.basicConfig(level=logging.DEBUG)

def test_can_instantiate():
    dir_list = os.listdir("models/cross-embedding")
    dir_list.sort()

    latest_checkpoint = dir_list[-1]

    model_name = f"models/cross-embedding/{latest_checkpoint}"

    cross_ranker = CrossRanker(model_name)
    assert cross_ranker != None

def test_get_scores_length():
    dir_list = os.listdir("models/cross-embedding")
    dir_list.sort()

    latest_checkpoint = dir_list[-1]

    model_name = f"models/cross-embedding/{latest_checkpoint}"

    cross_ranker = CrossRanker(model_name)
    scores = cross_ranker.get_scores(sample_query.loc[0, "tweet_text"], sample_collection)
    
    assert len(scores) == len(sample_collection)