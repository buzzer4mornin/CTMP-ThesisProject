import pandas as pd
import numpy as np
import time
import warnings
import os

# Ignore Depricated Pandas Warning
warnings.filterwarnings("ignore")

# Change to Current File Directory
os.chdir(os.path.dirname(__file__))

# Get Current File Directory
currdir = str(os.path.dirname(os.path.abspath(__file__)))

# ======================================================================================================================
# ======================================================================================================================
# ============================================== DATA DIMINISHER =======================================================

# ------- Diminished Movie -------
movie_df = pd.read_pickle("df_movie_UPDATED")

# [1, 47, 119, 358, 521]
movie_dim = movie_df.iloc[[1, 47, 119, 358, 521], :]
movie_dim.loc[:, "new_MOVIEID"] = range(1, 6)
movie_dim.to_pickle(currdir + '/df_movie_DIMINISHED')

# Print length of each document (TOTAL 5 docs)
# plots = movie_dim["MOVIEPLOT"].tolist()
# print([len(x) for x in plots])

# ------- Diminished User -------
user_df = pd.read_pickle("df_user_UPDATED")
user_dim = user_df.iloc[[1, 2, 3, 4, 5], :]
user_dim.to_pickle(currdir + '/df_user_DIMINISHED')


# ------- Diminished Rating -------
rating_df = pd.read_pickle("df_rating_UPDATED")
users = [i for i in list(range(1, 6)) for _ in range(5)]
movies = [i for _ in range(5) for i in list(range(1, 6))]
ratings = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0 , 1, 1, 0, 0, 1, 0, 0, 1, 1,  0, 0, 0, 0, 0]
rating_dim = pd.DataFrame(users, columns=["new_USERID"])
rating_dim["new_MOVIEID"] = movies
rating_dim["RATING"] = ratings
rating_dim.to_pickle(currdir + '/df_rating_DIMINISHED')
