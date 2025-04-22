import logging
import pandas as pd

from rankers.cross_embedding_ranker import CrossRanker

PATH_COLLECTION_DATA = '/home/guilherme/workspace/feup-ir-project/data/subtask_4b/subtask4b_collection_data.pkl'
PATH_QUERY_DATA = '/home/guilherme/workspace/feup-ir-project/data/subtask_4b/subtask4b_query_tweets.tsv'

df_query = pd.read_csv(PATH_QUERY_DATA, sep = '\t')
df_collection = pd.read_pickle(PATH_COLLECTION_DATA)


sample_collection = df_collection[:20]
sample_query = df_query[:5]

logging.basicConfig(level=logging.DEBUG)

def test_can_instantiate():
    cross_ranker = CrossRanker()
    assert cross_ranker != None

def test_get_scores_length():
    cross_ranker = CrossRanker()
    scores = cross_ranker.get_scores(sample_query.loc[0, "tweet_text"], sample_collection)
    
    assert len(scores) == len(sample_collection)