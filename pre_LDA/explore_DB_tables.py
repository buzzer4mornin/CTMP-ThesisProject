import pandas as pd
import numpy as np

# ======== Load plot_df ========
plot_df = pd.read_pickle("plot_df")
# Count of Movies without Plots ==> [870 / ~27k]
# print(plot_df[plot_df.MOVIEPLOT == 'N/A'].shape[0])
# Subselect Movies only with Plots
plot_df = plot_df[plot_df.MOVIEPLOT != 'N/A']

# ======== Load rating_df ========
rating_df = pd.read_pickle("rating_df")
rating_df["RATING"] = rating_df["RATING"].apply(lambda x: 1 if x >= 4 else 0)
#print(rating_df.shape[0])