import sys
import numpy as np
import numpy_indexed
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

def read_data(filename):
    wordids = list()
    wordcts = list()
    fp = open(filename, 'r')
    while True:
        line = fp.readline()
        # check end of file
        if len(line) < 1:
            break
        terms = line.split(' ')
        doc_length = int(terms[0])
        ids = np.zeros(doc_length, dtype=np.int32)
        cts = np.zeros(doc_length, dtype=np.int32)
        for j in range(1, doc_length + 1):
            term_count = terms[j].split(':')
            ids[j - 1] = int(term_count[0])
            cts[j - 1] = int(term_count[1])
        wordids.append(ids)
        wordcts.append(cts)
    fp.close()
    return wordids, wordcts


def read_setting(file_name):
    f = open(file_name, 'r')
    settings = f.readlines()
    f.close()
    sets = list()
    vals = list()
    for i in range(len(settings)):
        # print'%s\n'%(settings[i])
        if settings[i][0] == '#':
            continue
        set_val = settings[i].split(':')
        sets.append(set_val[0])
        vals.append(float(set_val[1]))
    ddict = dict(zip(sets, vals))
    ddict['num_docs'] = int(ddict['num_docs'])
    ddict['num_terms'] = int(ddict['num_terms'])
    ddict['num_topics'] = int(ddict['num_topics'])
    ddict['user_size'] = int(ddict['user_size'])
    ddict['tops'] = int(ddict['tops'])
    ddict['iter_infer'] = int(ddict['iter_infer'])
    ddict['iter_train'] = int(ddict['iter_train'])
    return ddict


def list_top(beta, tops):
    min_float = -sys.float_info.max
    num_tops = beta.shape[0]
    list_tops = list()
    for k in range(num_tops):
        top = list()
        arr = np.array(beta[k, :], copy=True)
        for t in range(tops):
            index = arr.argmax()
            top.append(index)
            arr[index] = min_float
        list_tops.append(top)
    return list_tops


def write_topic_top(list_tops, file_name):
    num_topics = len(list_tops)
    tops = len(list_tops[0])
    f = open(file_name, 'w')
    for k in range(num_topics):
        for j in range(tops - 1):
            f.write('%d ' % (list_tops[k][j]))
        f.write('%d\n' % (list_tops[k][tops - 1]))
    f.close()


def write_setting(ddict, file_name):
    keys = list(ddict.keys())
    vals = list(ddict.values())
    f = open(file_name, 'w')
    for i in range(len(keys)):
        f.write(f"{keys[i]}: {vals[i]}\n")
    f.close()


def print_diff_list_tops(list_tops, prev_list_tops, i):
    if i == 0:
        num_topics = len(list_tops)
        tops = len(list_tops[0])
        list_tops = np.array(list_tops)
        init = np.negative(np.ones([num_topics, tops], dtype=int))
        diff = init == list_tops
        diff_count = np.count_nonzero(diff)
        print("Difference:", diff_count)
    else:
        list_tops = np.array(list_tops)
        diff = prev_list_tops == list_tops
        diff_count = np.count_nonzero(diff)
        print("Difference:", diff_count)


def write_file(model_folder, list_tops, algo):
    list_tops_file_name = f'{model_folder}/list_tops.txt'
    write_topic_top(list_tops, list_tops_file_name)
    def file_locator(x): return f'{model_folder}/{str(x)}'
    files = [attr for attr in dir(algo) if attr in ["theta", "mu", "phi", "shp", "rte", "beta"]]
    for i in files:
        if i != "phi":
            np.save(file_locator(i), getattr(algo, i))
        else:
            #pass
            # TODO: uncomment below in final version
            with open(f"{model_folder}/phi.pkl", "wb") as f:
                pickle.dump(getattr(algo, i), f, protocol=4)


def get_rating_group(rating_group_file, k_cv):
    skf = StratifiedKFold(n_splits=k_cv, shuffle=True, random_state=42)
    ratings = np.array(pd.read_pickle(rating_group_file))

    #TODO: After doing above, then run below to get 1 fold of kfold, and then train on that fold
    ratings_one = ratings[np.where(ratings[:, 2] == 1)]
    users = ratings_one[:, 0]
    ratings_one, users = pd.DataFrame(ratings_one), pd.DataFrame(users)

    for train_index, test_index in skf.split(ratings_one, users):
        ratings_one_train = np.array(ratings_one.loc[train_index, :])
        ratings_one_test = np.array(ratings_one.loc[test_index, :])
        break

    all_mov_ids = np.unique(ratings[:, 1])
    all_usr_ids = np.unique(ratings[:, 0])

    # --------------- TRAIN ---------------
    temp = ratings_one_train[:, [0, 1]]
    # Creating GroupForUser dictionary
    temp_sorted = temp[np.argsort(temp[:, 0])]
    usr_id = np.unique(temp_sorted[:, 0])
    mov_id = numpy_indexed.group_by(temp_sorted[:, 0]).split(temp_sorted[:, 1])
    rating_GroupForUser_train = dict(zip(usr_id, mov_id))
    diff = np.setdiff1d(all_usr_ids, usr_id)
    for i in diff: rating_GroupForUser_train[i] = []

    # Creating GroupForMovie dictionary
    temp_sorted = temp[np.argsort(temp[:, 1])]
    mov_id = np.unique(temp_sorted[:, 1])
    usr_id = numpy_indexed.group_by(temp_sorted[:, 1]).split(temp_sorted[:, 0])
    rating_GroupForMovie_train = dict(zip(mov_id, usr_id))
    diff = np.setdiff1d(all_mov_ids, mov_id)
    for i in diff: rating_GroupForMovie_train[i] = []

    # --------------- TEST ---------------
    temp = ratings_one_test[:, [0, 1]]
    # Creating GroupForUser dictionary
    temp_sorted = temp[np.argsort(temp[:, 0])]
    usr_id = np.unique(temp_sorted[:, 0])
    mov_id = numpy_indexed.group_by(temp_sorted[:, 0]).split(temp_sorted[:, 1])
    rating_GroupForUser_test = dict(zip(usr_id, mov_id))
    diff = np.setdiff1d(all_usr_ids, usr_id)
    for i in diff: rating_GroupForUser_test[i] = []

    # Creating GroupForMovie dictionary
    temp_sorted = temp[np.argsort(temp[:, 1])]
    mov_id = np.unique(temp_sorted[:, 1])
    usr_id = numpy_indexed.group_by(temp_sorted[:, 1]).split(temp_sorted[:, 0])
    rating_GroupForMovie_test = dict(zip(mov_id, usr_id))
    diff = np.setdiff1d(all_mov_ids, mov_id)
    for i in diff: rating_GroupForMovie_test[i] = []

    return rating_GroupForUser_train, rating_GroupForMovie_train, rating_GroupForUser_test, rating_GroupForMovie_test
