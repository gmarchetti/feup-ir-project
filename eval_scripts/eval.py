def get_performance_mrr(data, col_gold, col_pred, list_k = [1, 5, 10]):
    d_performance = {}
    for k in list_k:
        data["in_topx"] = data.apply(lambda x: (1/([i for i in x[col_pred][:k]].index(x[col_gold]) + 1) if x[col_gold] in [i for i in x[col_pred][:k]] else 0), axis=1)
        d_performance[k] = data["in_topx"].mean()
    return d_performance

def get_avg_gold_in_pred(data, col_gold, col_pred, list_k = [1, 5, 10]):
    d_performance = {}
    for k in list_k:
        data["in_topx"] = data.apply(lambda x: (1 if x[col_gold] in [i for i in x[col_pred][:k]] else 0), axis=1)
        d_performance[k] = data["in_topx"].mean()
    return d_performance

def create_pred_file(query_set, prediction_columns):
    query_set['preds'] = query_set[prediction_columns].apply(lambda x: x[:5])
    query_set[['post_id', 'preds']].to_csv('predictions.tsv', index=None, sep='\t')
    