import pandas as pd
import numpy as np
import time
import os

# ======== Load user_df ========
user_df = pd.read_pickle("user_df")


# ======== Load movie_df =========
movie_df = pd.read_pickle("movie_df")
# Count of Movies without Plots ==> [870 / ~27k]
# print(movie_df[movie_df.MOVIEPLOT == 'N/A'].shape[0])
# Subselect Movies only with Plots
print("unique MOVIEIDs before dropping NAs [movie]   -- ", movie_df.shape[0])
movie_df = movie_df[movie_df.MOVIEPLOT != 'N/A']
# print(movie_df.head())

# ======== Load rating_df ========
rating_df = pd.read_pickle("rating_df")
rating_df["RATING"] = rating_df["RATING"].apply(lambda x: 1 if x >= 4 else 0)
# print(rating_df.head())

# ========== Clean movie_df & rating_df ===========
movie_unique = movie_df["MOVIEID"]
rating_unique = rating_df["MOVIEID"].drop_duplicates()
joint = pd.merge(movie_unique, rating_unique, how='inner', on='MOVIEID')
movie_df = pd.merge(movie_df, joint, how='inner', on='MOVIEID')
rating_df = pd.merge(rating_df, joint, how='inner', on='MOVIEID')

print("unique MOVIEIDs after dropping NAs  [movie]   -- ", movie_unique.shape[0])
print("unique MOVIEIDs in 20M               [rating] -- ", rating_unique.shape[0])
print("unique MOVIEIDS jointly by  [movie]vs[rating] -- ", joint.shape[0])

print("========================================================")
print("DONE... TEST: ", pd.merge(rating_df, movie_df, how='inner', on="MOVIEID")["MOVIEID"].nunique())
print("========================================================")

print("SAVING... cleaned movie_df & rating_df")
movie_df.to_pickle((str(os.path.dirname(os.path.abspath(__file__))) + '/movie_df'))
rating_df.to_pickle((str(os.path.dirname(os.path.abspath(__file__))) + '/rating_df'))

