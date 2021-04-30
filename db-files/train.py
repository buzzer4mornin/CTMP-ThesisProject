import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np



'''import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")
skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)

df = np.array([
    [0, 4, 1], [0, 2, 1], [0, 1, 1], [0, 5, 1], [0, 6, 1], [0, 3, 1],
    [1, 0, 1], [1, 4, 1], [1, 7, 1], [1, 8, 1], [1, 9, 1], [1, 2, 1], [1, 6, 1], [1, 5, 1]
])



usrs = df[:, 0]

df = pd.DataFrame(df)
usrs = pd.DataFrame(usrs)

for train_index, test_index in skf.split(df, usrs):
    train = df.loc[train_index, :]
    test = df.loc[test_index, :]
    print(np.array(train))
    print("----")
    print(np.array(test))
    print("@@@@@@@@@@@@@@@@@@@")'''


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


# Read data
with open(f"./rating_GroupForUser_test.pkl", "rb") as f:
    rating_GroupForUser_test = pickle.load(f)

with open(f"./rating_GroupForMovie_test.pkl", "rb") as f:
    rating_GroupForMovie_test = pickle.load(f)

# Read data
with open(f"./rating_GroupForUser_train.pkl", "rb") as f:
    rating_GroupForUser_train = pickle.load(f)

with open(f"./rating_GroupForMovie_train.pkl", "rb") as f:
    rating_GroupForMovie_train = pickle.load(f)


# ---------- Users Proportion Test ----------
u_test = 0
for key in rating_GroupForUser_test:
    u_test += len(rating_GroupForUser_test[key])

u_train = 0
for key in rating_GroupForUser_train:
    u_train += len(rating_GroupForUser_train[key])

# Correct is 0.2 --> 5-fold cross validation
# print(u_test/(u_test + u_train))

# ---------- Movies Test ----------
m_test = 0
for key in rating_GroupForMovie_test:
    m_test += len(rating_GroupForMovie_test[key])

m_train = 0
for key in rating_GroupForMovie_train:
    m_train += len(rating_GroupForMovie_train[key])

# Correct is 0.2 --> 5-fold cross validation
# print(m_test/(m_test + m_train))

# =================================================

less_test = 0
for key in rating_GroupForUser_test:
    if len(rating_GroupForUser_test[key]) <= 5:
        less_test += 1

less_train = 0
for key in rating_GroupForUser_train:
    if len(rating_GroupForUser_train[key]) <= 5:
        less_train += 1


# Badly distributed
print(less_test/len(rating_GroupForUser_test))
print(less_train/len(rating_GroupForUser_train))

