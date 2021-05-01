import sys
import numpy as np
import numpy_indexed
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")


# ratings = np.array(np.load("./df_rating_UPDATED", allow_pickle=True))
# ratings_one = ratings[np.where(ratings[:, 2] == 1)]
#
# all_mov_ids = np.unique(ratings[:, 1])
# all_usr_ids = np.unique(ratings[:, 0])
#
# # --------------- TRAIN ---------------
# temp = ratings_one[:, [0, 1]]
# # Creating GroupForUser dictionary
# temp_sorted = temp[np.argsort(temp[:, 0])]
# usr_id = np.unique(temp_sorted[:, 0])
# mov_id = numpy_indexed.group_by(temp_sorted[:, 0]).split(temp_sorted[:, 1])
# rating_GroupForUser = dict(zip(usr_id, mov_id))
# diff = np.setdiff1d(all_usr_ids, usr_id)
# for i in diff: rating_GroupForUser[i] = []
#
# # Creating GroupForMovie dictionary
# temp_sorted = temp[np.argsort(temp[:, 1])]
# mov_id = np.unique(temp_sorted[:, 1])
# usr_id = numpy_indexed.group_by(temp_sorted[:, 1]).split(temp_sorted[:, 0])
# rating_GroupForMovie = dict(zip(mov_id, usr_id))
# diff = np.setdiff1d(all_mov_ids, mov_id)
# for i in diff: rating_GroupForMovie[i] = []
#
# with open(f"rating_GroupForUser.pkl", "wb") as f:
#     pickle.dump(rating_GroupForUser, f)
# with open(f"rating_GroupForMovie.pkl", "wb") as f:
#     pickle.dump(rating_GroupForMovie, f)


# ======================================================================================================================

def get_rating_group(rating_group_file, k_cv, args):
    skf = StratifiedKFold(n_splits=k_cv, shuffle=True, random_state=args.seed)
    ratings = np.array(pd.read_pickle(rating_group_file))

    ratings_one = ratings[np.where(ratings[:, 2] == 1)]
    users = ratings_one[:, 0]
    ratings_one, users = pd.DataFrame(ratings_one), pd.DataFrame(users)

    train_folds = []
    test_folds = []
    for train_index, test_index in skf.split(ratings_one, users):
        ratings_one_train = np.array(ratings_one.loc[train_index, :])
        ratings_one_test = np.array(ratings_one.loc[test_index, :])

        all_mov_ids = np.unique(ratings[:, 1])
        all_usr_ids = np.unique(ratings[:, 0])

        # --------------- TRAIN ---------------
        temp = ratings_one_train[:, [0, 1]]
        # Creating GroupForUser dictionary for TRAIN Dataset
        temp_sorted = temp[np.argsort(temp[:, 0])]
        usr_id = np.unique(temp_sorted[:, 0])
        mov_id = numpy_indexed.group_by(temp_sorted[:, 0]).split(temp_sorted[:, 1])
        rating_GroupForUser_train = dict(zip(usr_id, mov_id))
        diff = np.setdiff1d(all_usr_ids, usr_id)
        for i in diff: rating_GroupForUser_train[i] = []

        # Creating GroupForMovie dictionary for TRAIN Dataset
        temp_sorted = temp[np.argsort(temp[:, 1])]
        mov_id = np.unique(temp_sorted[:, 1])
        usr_id = numpy_indexed.group_by(temp_sorted[:, 1]).split(temp_sorted[:, 0])
        rating_GroupForMovie_train = dict(zip(mov_id, usr_id))
        diff = np.setdiff1d(all_mov_ids, mov_id)
        for i in diff: rating_GroupForMovie_train[i] = []

        train_folds.append([rating_GroupForUser_train, rating_GroupForMovie_train])

        # --------------- TEST ---------------
        temp = ratings_one_test[:, [0, 1]]
        # Creating GroupForUser dictionary for TEST Dataset
        temp_sorted = temp[np.argsort(temp[:, 0])]
        usr_id = np.unique(temp_sorted[:, 0])
        mov_id = numpy_indexed.group_by(temp_sorted[:, 0]).split(temp_sorted[:, 1])
        rating_GroupForUser_test = dict(zip(usr_id, mov_id))
        diff = np.setdiff1d(all_usr_ids, usr_id)
        for i in diff: rating_GroupForUser_test[i] = []

        # Creating GroupForMovie dictionary for TEST Dataset
        temp_sorted = temp[np.argsort(temp[:, 1])]
        mov_id = np.unique(temp_sorted[:, 1])
        usr_id = numpy_indexed.group_by(temp_sorted[:, 1]).split(temp_sorted[:, 0])
        rating_GroupForMovie_test = dict(zip(mov_id, usr_id))
        diff = np.setdiff1d(all_mov_ids, mov_id)
        for i in diff: rating_GroupForMovie_test[i] = []

        test_folds.append([rating_GroupForUser_test, rating_GroupForMovie_test])

    return train_folds, test_folds


    # less_test = 0
    # for key in rating_GroupForUser_test:
    #     if len(rating_GroupForUser_test[key]) <= 5:
    #         less_test += 1
    #
    # less_train = 0
    # for key in rating_GroupForUser_train:
    #     if len(rating_GroupForUser_train[key]) <= 5:
    #         less_train += 1
    #
    # # Badly distributed
    # print("----")
    # print(less_test/len(rating_GroupForUser_test))
    # print(less_train/len(rating_GroupForUser_train))
    # print("----")

