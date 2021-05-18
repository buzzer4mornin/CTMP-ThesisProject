import pandas as pd
import numpy as np
import time
import os

# Change to Current File Directory
os.chdir(os.path.dirname(__file__))

# Get Current File Directory
currdir = str(os.path.dirname(os.path.abspath(__file__)))

# ======================================================================================================================
# ======================================================================================================================
# ============================================== DATA CLEANER ==========================================================
'''  
Clean raw data(coming from DataBase) 

Inputs should be:
df_user           : 138493 rows, each is unique user_id 
df_movie          : 27278 rows, each is unique movie_id 
df_rating         : 20000263 rows, each is unique movie_id + user_id combination 

Outputs should be: 
df_user_CLEANED   : 138493 rows, each is random unique user_id 
df_movie_CLEANED  : 25900 rows, each is unique random movie_id 
df_rating_CLEANED : 19994181 rows, each is unique random movie_id + user_id combination '''
print("\n********************* DATA CLEANER *********************")

# ======== Load df_user ========
user_df = pd.read_pickle("./original-files/df_user_NFLX")

# ======== Load df_movie =========
movie_df = pd.read_pickle("./original-files/df_movie_NFLX")

# Count of Movies without Plots ==> [870 / ~27k]
# print(df_movie[df_movie.MOVIEPLOT == 'N/A'].shape[0])
# Subselect Movies only with Plots
print("unique MOVIEIDs before dropping NAs   [movie] --", movie_df.shape[0])
movie_df = movie_df[movie_df.MOVIEPLOT != 'N/A']
# print(df_movie.head())


# ======== Load df_rating ========
rating_df = pd.read_pickle("./original-files/df_rating_NFLX")
rating_df["RATING"] = rating_df["RATING"].apply(lambda x: 1 if x >= 4 else 0)
# print(rating_df.head())


# ====== Clean df_movie & df_rating ======
movie_unique = movie_df["MOVIEID"]
rating_unique = rating_df["MOVIEID"].drop_duplicates()
joint = pd.merge(movie_unique, rating_unique, how='inner', on='MOVIEID')
movie_df = pd.merge(movie_df, joint, how='inner', on='MOVIEID')
rating_df = pd.merge(rating_df, joint, how='inner', on='MOVIEID')

print("unique MOVIEIDs after dropping NAs    [movie] --", movie_unique.shape[0])
print("unique MOVIEIDs in 20M               [rating] --", rating_unique.shape[0])
print("unique MOVIEIDS jointly by   [movie & rating] --", joint.shape[0])

print("\nDONE... FINAL TEST: ", pd.merge(rating_df, movie_df, how='inner', on="MOVIEID")["MOVIEID"].nunique())
print("SAVING... CLEANED -> | df_user_CLEANED | df_movie_CLEANED | df_rating_CLEANED | \n")
user_df.to_pickle(currdir + '/df_user_NFLX_CLEANED')
movie_df.to_pickle(currdir + '/df_movie_NFLX_CLEANED')
rating_df.to_pickle(currdir + '/df_rating_NFLX_CLEANED')

# ======================================================================================================================
# ======================================================================================================================
# ============================================== DATA UPDATER ==========================================================
'''
Update ids in cleaned data(coming from above), 
Where user and movie ids are consistent - in range of respective table's row count.

Inputs should be:
df_user_CLEANED: random unique ids - total length of 138493
df_movie_CLEANED: random unique ids - total length of 25900
df_rating_CLEANED: contains random user and movie ids - total length of 19994181

Outputs should be:
df_user_UPDATED: consistent in range(138493)
df_movie_UPDATED: consistent in range(25900)
df_rating_UPDATED: contains updated user and movie ids - total length of 19994181

DEEPNOTE: Size of input and output tables are same. Only ids are converted into consistent ones. 
'''

print("********************* DATA UPDATER *********************")

# read tables
user_df = pd.read_pickle("processed-files/df_user_NFLX_CLEANED")
movie_df = pd.read_pickle("processed-files/df_movie_NFLX_CLEANED")
rating_df = pd.read_pickle("processed-files/df_rating_NFLX_CLEANED")

# add consistent user id column
user_df["new_USERID"] = np.arange(len(user_df))
# print(user_df)


# add consistent movie id column
movie_df = movie_df.drop(["TT"], axis=1)
movie_df["new_MOVIEID"] = np.arange(len(movie_df))
# print(movie_df)


# convert df into np
user_np = np.array(user_df)
movie_np = np.array(movie_df[["MOVIEID", "new_MOVIEID"]])
rating_np = np.array(rating_df)

# Save Updated User and Movie tables
print("SAVING... updated df_user_UPDATED & df_movie_UPDATED")
user_df = user_df.drop(["USERID"], axis=1)
movie_df = movie_df.drop(["MOVIEID"], axis=1)
user_df.to_pickle(currdir + '/df_user_NFLX_UPDATED')
movie_df.to_pickle(currdir + '/df_movie_NFLX_UPDATED')

# Save Updated Rating table
print(rating_np)
s = time.time()
user_np = user_np[:, 0].reshape(1, -1).flatten()
movie_np = movie_np[:, 0].reshape(1, -1).flatten()
first = lambda x: np.where(user_np == x)[0][0]
second = lambda x: np.where(movie_np == x)[0][0]
v_user = np.vectorize(first)
v_movie = np.vectorize(second)
rating_np[:, 0] = v_user(rating_np[:, 0])
rating_np[:, 1] = v_movie(rating_np[:, 1])
e = time.time()
print("SAVING... updated df_rating_UPDATED || exec.time : {:.4g} min".format((e - s) / 60))
print(rating_np)
rating_df = pd.DataFrame(rating_np)
rating_df.columns = ["new_USERID", "new_MOVIEID", "RATING"]
rating_df.to_pickle(currdir + '/df_rating_NFLX_UPDATED')
