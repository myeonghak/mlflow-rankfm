### Import Required Packages and Set Options

import os
import sys
# import git

import numpy as np
import numba as nb
import pandas as pd

import warnings
# warnings.filterwarnings("ignore", category=nb.NumbaPerformanceWarning)

#### Dynamically Re-Load all Package Modules

import mlflow
from mlflow import log_metric, log_param, log_artifacts



from rankfm.rankfm import RankFM
from rankfm.evaluation import hit_rate, reciprocal_rank, discounted_cumulative_gain, precision, recall, diversity

import neptune
import neptune_mlflow

neptune.init(project_qualified_name='myeonghak/example')


#Load data
path = "/Users/a420777/Desktop/hcc/gnn/datasets/movielens/"

#Ratings
interactions = pd.read_csv(path+'ratings.dat', sep='::', header=None, engine='python')
interactions.columns = ['user_id','item_id','rating','timestamp']
interactions = interactions.drop('timestamp', axis=1)

#Movies
item_names = pd.read_csv(path+'movies.dat', sep='::', header=None, engine='python')
item_names.columns = ['item_id','title','genres']

#Users
user_features = pd.read_csv(path+'users.dat', sep='::', header=None, engine='python')
user_features.columns = ['user_id','gender','age','occupation','zip-code']
user_features = user_features.drop('zip-code', axis=1)

# interactions = pd.read_csv(os.path.join(data_path, 'ML_1M_RATINGS.csv'))
interactions.head()

# user_features = pd.read_csv(os.path.join(data_path, 'ML_1M_USERS.csv'))
user_features.occupation=user_features.occupation.astype("object")
user_features.age=user_features.age.astype("object")

user_features=pd.get_dummies(user_features)

import collections

split_series=item_names.genres.str.split("|").apply(lambda x: x)

split_series_dict=split_series.apply(collections.Counter)
multi_hot=pd.DataFrame.from_records(split_series_dict).fillna(value=0)



item_features=pd.concat([item_names.item_id, multi_hot], axis=1)


#### Check Matrix/Vector Dimensions

unique_users = interactions.user_id.nunique()
unique_items = interactions.item_id.nunique()

print("interactions shape: {}".format(interactions.shape))
print("interactions unique users: {}".format(interactions.user_id.nunique()))
print("interactions unique items: {}".format(interactions.item_id.nunique()))

print("user features users:", interactions.user_id.nunique())
print("item features items:", interactions.item_id.nunique())

#### Evaluate Interaction Matrix Sparsity

sparsity = 1 - (len(interactions) / (unique_users * unique_items))
print("interaction matrix sparsity: {}%".format(round(100 * sparsity, 1)))




np.random.seed(1492)
interactions['random'] = np.random.random(size=len(interactions))
test_pct = 0.25

train_mask = interactions['random'] <  (1 - test_pct)
valid_mask = interactions['random'] >= (1 - test_pct)

interactions_train = interactions[train_mask][['user_id', 'item_id']]
interactions_valid = interactions[valid_mask][['user_id', 'item_id']]

train_users = np.sort(interactions_train.user_id.unique())
valid_users = np.sort(interactions_valid.user_id.unique())
cold_start_users = set(valid_users) - set(train_users)

train_items = np.sort(interactions_train.item_id.unique())
valid_items = np.sort(interactions_valid.item_id.unique())
cold_start_items = set(valid_items) - set(train_items)

print("train shape: {}".format(interactions_train.shape))
print("valid shape: {}".format(interactions_valid.shape))

print("train users: {}".format(len(train_users)))
print("valid users: {}".format(len(valid_users)))
print("cold-start users: {}".format(cold_start_users))

print("train items: {}".format(len(train_items)))
print("valid items: {}".format(len(valid_items)))
print("cold-start items: {}".format(cold_start_items))

user_features = user_features[user_features.user_id.isin(train_users)]
item_features = item_features[item_features.item_id.isin(train_items)]
user_features.shape, item_features.shape

### Fit the Model on the Training Data and Evaluate Out-of-Sample Performance Metrics

#### Initialize the Model Object

num_factors=20

log_param("factor",num_factors)

model = RankFM(factors=num_factors, loss='warp', max_samples=20, alpha=0.01, sigma=0.1, learning_rate=0.10, learning_schedule='invscaling')


model.fit(interactions_train, epochs=1, verbose=True)

mlflow.pyfunc.log_model(model, "model")

valid_scores = model.predict(interactions_valid, cold_start='nan') 



print(valid_scores.shape)
pd.Series(valid_scores).describe()

valid_recommendations = model.recommend(valid_users, n_items=10, filter_previous=True, cold_start='nan')
valid_recommendations.head()

#### Evaluate Model Performance on the Validation Data

k = 10

##### Generate Pure-Popularity Baselines

most_popular = interactions_train.groupby('item_id')['user_id'].count().sort_values(ascending=False)[:k]
most_popular

test_user_items = interactions_valid.groupby('user_id')['item_id'].apply(set).to_dict()
test_user_items = {key: val for key, val in test_user_items.items() if key in set(train_users)}

base_hrt = np.mean([int(len(set(most_popular.index) & set(val)) > 0)                       for key, val in test_user_items.items()])
base_pre = np.mean([len(set(most_popular.index) & set(val)) / len(set(most_popular.index)) for key, val in test_user_items.items()])
base_rec = np.mean([len(set(most_popular.index) & set(val)) / len(set(val))                for key, val in test_user_items.items()])


log_metric("hit_rate", base_hrt)



print("number of test users: {}".format(len(test_user_items)))
print("baseline hit rate: {:.3f}".format(base_hrt))
print("baseline precision: {:.3f}".format(base_pre))
print("baseline recall: {:.3f}".format(base_rec))



model_hit_rate = hit_rate(model, interactions_valid, k=k)
model_reciprocal_rank = reciprocal_rank(model, interactions_valid, k=k)
model_dcg = discounted_cumulative_gain(model, interactions_valid, k=k)
model_precision = precision(model, interactions_valid, k=k)
model_recall = recall(model, interactions_valid, k=k)



print("hit_rate: {:.3f}".format(model_hit_rate))
print("reciprocal_rank: {:.3f}".format(model_reciprocal_rank))
print("dcg: {:.3f}".format(model_dcg, 3))
print("precision: {:.3f}".format(model_precision))
print("recall: {:.3f}".format(model_recall))



# recommendation_diversity = diversity(model, interactions_valid, k=k)
# recommendation_diversity.head(10)

# top_items = pd.merge(item_names, recommendation_diversity, on='item_id', how='inner')
# top_items = top_items.set_index('item_id').loc[recommendation_diversity.item_id].reset_index()
# top_items = top_items[['item_id', 'cnt_users', 'pct_users', 'title', 'genres']]
# top_items.head(10)

# coverage = np.mean(recommendation_diversity['cnt_users'] > 0)
# print("percentage of items recommended to at least one user: {:.3f}".format(coverage))

# nonzero_users = recommendation_diversity[recommendation_diversity.cnt_users > 0]
# entropy = -np.sum(nonzero_users['pct_users'] * np.log2(nonzero_users['pct_users']))
# print("entropy value of recommended items: {:.3f}".format(entropy))

# N = 50
# fig, axes = plt.subplots(1, 1, figsize=[16, 4])

# topN = recommendation_diversity.iloc[:N, :]
# axes.bar(topN.index.values + 1, topN.pct_users, width=1, edgecolor='black', alpha=0.75)
# axes.set(xlabel='Item Rank', ylabel='Percentage of Users', title='Percentage of Users Recommended by Item Rank')

