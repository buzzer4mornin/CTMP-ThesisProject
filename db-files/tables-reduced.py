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
# ============================================== DATA REDUCER ==========================================================
'''  
Get datasample for CTMP by REDUCING the UPDATED data 

Inputs should be:
df_user_UPDATED: consistent in range(138493)
df_movie_UPDATED: consistent in range(25900)
df_rating_UPDATED: contains updated user and movie ids - total length of 19994181

Outputs should be:
df_user_REDUCED   : consistent in range(1493) 
df_movie_REDUCED  : consistent in range(1000)
df_rating_REDUCED : 1470 rows, each is unique consistent movie_id + user_id combination '''

# read tables
user_df = pd.read_pickle("df_user_UPDATED")
movie_df = pd.read_pickle("df_movie_UPDATED")
rating_df = pd.read_pickle("df_rating_UPDATED")

# take sample movie ids and rename column
user_df = user_df.iloc[0:10000, :]
user_df.rename(columns={'new_USERID': 'sample_USERID'}, inplace=True)

# take sample movie ids and rename column
movie_df = movie_df.iloc[0:1000, :]
movie_df.rename(columns={'new_MOVIEID': 'sample_MOVIEID'}, inplace=True)


# extract ratings corresponding to sample movie/user ids
sample_user_ids = np.arange(10000)
sample_movie_ids = np.arange(1000)
rating_np = np.array(rating_df)
mask_user = np.isin(rating_np[:, 0], sample_user_ids)
rating_np = rating_np[mask_user]
mask_movie = np.isin(rating_np[:, 1], sample_movie_ids)
rating_np = rating_np[mask_movie]

# REDUCE user ids and movie ids according to extracted ratings
user_reduced_ids = np.unique(rating_np[:, 0])
movie_reduced_ids = np.unique(rating_np[:, 1])


# Save REDUCED User table
print("SAVING... updated df_user_REDUCED")
user_np = np.array(user_df)
mask_reduced = np.isin(user_np[:, 0], user_reduced_ids)
user_df = pd.DataFrame(user_np[mask_reduced])
user_df.columns = ["reduced_USERID"]
user_df.to_pickle(currdir + '/df_user_REDUCED')


# Save REDUCED Movie table
print("SAVING... updated df_movie_REDUCED")
movie_np = np.array(movie_df)
mask_reduced = np.isin(movie_np[:, 1], movie_reduced_ids)
movie_df = pd.DataFrame(movie_np[mask_reduced])
movie_df.columns = ["MOVIEPLOT", "reduced_MOVIEID"]
movie_df.to_pickle(currdir + '/df_movie_REDUCED')


# Save REDUCED Rating table
print("SAVING... updated df_rating_REDUCED")
rating_df = pd.DataFrame(rating_np)
rating_df.columns = ["reduced_USERID", "reduced_MOVIEID", "RATING"]
rating_df.to_pickle(currdir + '/df_rating_REDUCED')
