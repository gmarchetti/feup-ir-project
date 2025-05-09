import logging
import pandas as pd
from os import listdir
from rankers.pairwise_ranker import PairwiseRanker

PATH_COLLECTION_DATA = 'data/subtask_4b/subtask4b_collection_data.pkl'
PATH_QUERY_DATA = 'data/dev-tweets/subtask4b_query_tweets_dev.tsv'

df_query = pd.read_csv(PATH_QUERY_DATA, sep = '\t')
df_collection = pd.read_pickle(PATH_COLLECTION_DATA)

pre_sampled_docids = ['hg3xpej0', 'styavbvi', '3qvh482o', 'rthsl7a9', '1adt71pk']

sample_collection = df_collection[df_collection['cord_uid'].isin(pre_sampled_docids)]
sample_query = df_query[:5]

logging.basicConfig(level=logging.DEBUG)

def test_can_instantiate():
    base_name = "models/pairwise-classifier"
    dir_list = listdir(base_name)
    dir_list.sort()

    latest_checkpoint = dir_list[-1]

    model_name = f"{base_name}/{latest_checkpoint}"

    pairwise_ranker = PairwiseRanker(model_name)
    assert pairwise_ranker != None

def test_get_higher_rel_prob():    
    base_name = "models/pairwise-classifier-large"
    dir_list = listdir(base_name)
    dir_list.sort()

    latest_checkpoint = dir_list[-1]

    model_name = f"{base_name}/{latest_checkpoint}"

    pairwise_ranker = PairwiseRanker(model_name)

    score = pairwise_ranker.get_higher_rel_prob(sample_query.loc[0, "tweet_text"], sample_collection.iloc[0], sample_collection.iloc[1], {})
    assert type(score) is float

def test_rank_avg_prob():
    collection_as_dict = sample_collection.to_dict('index')
    doc_index_list = list(collection_as_dict.keys())
    
    presorted_uids = [collection_as_dict[index]["cord_uid"] for index in doc_index_list]
    logging.debug(presorted_uids)
    
    base_name = "models/pairwise-classifier-large"
    dir_list = listdir(base_name)
    dir_list.sort()

    latest_checkpoint = dir_list[-1]

    model_name = f"{base_name}/{latest_checkpoint}"

    pairwise_ranker = PairwiseRanker(model_name)
    sorted_uids = pairwise_ranker.rank_avg_prob(sample_query.loc[0, "tweet_text"], sample_collection)
    
    logging.debug(sorted_uids)
    assert len(pre_sampled_docids) == len(sorted_uids)