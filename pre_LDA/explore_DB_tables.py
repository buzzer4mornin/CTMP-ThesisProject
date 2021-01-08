import pandas as pd
import numpy as np
import time

# ======== Load plot_df =========
plot_df = pd.read_pickle("plot_df")
# Count of Movies without Plots ==> [870 / ~27k]
# print(plot_df[plot_df.MOVIEPLOT == 'N/A'].shape[0])
# Subselect Movies only with Plots
print("unique MOVIEIDs before dropping NAs [plot]   -- ", plot_df.shape[0])
plot_df = plot_df[plot_df.MOVIEPLOT != 'N/A']
# print(plot_df.head())


# ======== Load rating_df ========
rating_df = pd.read_pickle("rating_df")
rating_df["RATING"] = rating_df["RATING"].apply(lambda x: 1 if x >= 4 else 0)
# print(rating_df.head())


# ============ TRAINING =========
plot_unique = plot_df["MOVIEID"]
rating_unique = rating_df["MOVIEID"].drop_duplicates()
joint = pd.merge(plot_unique, rating_unique, how='inner', on='MOVIEID')
plot_final = pd.merge(plot_df, joint, how='inner', on='MOVIEID')
rating_final = pd.merge(rating_df, joint, how='inner', on='MOVIEID')

print("unique MOVIEIDs after dropping NAs  [plot]   -- ", plot_unique.shape[0])
print("unique MOVIEIDs in 20M              [rating] -- ", rating_unique.shape[0])
print("unique MOVIEIDS jointly by  [plot]vs[rating] -- ", joint.shape[0])
print("========================================================")
print("DONE... TEST: ", pd.merge(rating_final, plot_final, how='inner', on="MOVIEID")["MOVIEID"].nunique())
