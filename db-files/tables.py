import pandas as pd
import numpy as np
import time
import os

# Change to Current File Directory
os.chdir(os.path.dirname(__file__))

# Get Current File Directory
currdir = str(os.path.dirname(os.path.abspath(__file__)))

# ======== Load df_user ========
user_df = pd.read_pickle("df_user")


# ======== Load df_movie =========
movie_df = pd.read_pickle("df_movie")
# Count of Movies without Plots ==> [870 / ~27k]
# print(df_movie[df_movie.MOVIEPLOT == 'N/A'].shape[0])
# Subselect Movies only with Plots
print("unique MOVIEIDs before dropping NAs [movie]   -- ", movie_df.shape[0])
movie_df = movie_df[movie_df.MOVIEPLOT != 'N/A']
# print(df_movie.head())


# ======== Load df_rating ========
rating_df = pd.read_pickle("df_rating")
rating_df["RATING"] = rating_df["RATING"].apply(lambda x: 1 if x >= 4 else 0)

# print(df_rating.head())


# ====== Clean df_movie & df_rating ======
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

print("SAVING... cleaned df_movie & df_rating")
movie_df.to_pickle(currdir + '/df_movie')
rating_df.to_pickle(currdir + '/df_rating')

#TODO: create __main__ for prevention of double filtering of ratings table