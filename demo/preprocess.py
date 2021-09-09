import numpy as np
import pandas as pd


path = "../dataset/"

    
def preprocess_data():
    #Ratings
    interactions = pd.read_csv(path+'ratings.dat', sep='::', header=None, engine='python', encoding = "ISO-8859-1")
    interactions.columns = ['user_id','item_id','rating','timestamp']
    interactions = interactions.drop('timestamp', axis=1)

    #Movies
    item_names = pd.read_csv(path+'movies.dat', sep='::', header=None, engine='python', encoding = "ISO-8859-1")
    item_names.columns = ['item_id','title','genres']

    #Users
    user_features = pd.read_csv(path+'users.dat', sep='::', header=None, engine='python', encoding = "ISO-8859-1")
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

#     user_features = user_features[user_features.user_id.isin(train_users)]
#     item_features = item_features[item_features.item_id.isin(train_items)]
    

    return interactions_train, interactions_valid


if __name__=="__main__":
    run()
    
