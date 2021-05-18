#!/usr/bin/python
# -*- coding: utf-8 -*-


import shutil
import sys
import time
import pickle
import pandas as pd
from math import floor
import os
from CTMP import MyCTMP
from LDA import MyLDA
from Evaluation import MyEvaluation

sys.path.insert(0, './common')
import utilities


# ------------ RUN in terminal ------------
# --> python ./model/run_model.py ctmp original 5
# --> python ./model/run_model.py lda original 5

# --> python ./model/run_model.py ctmp reduced 5
# --> python ./model/run_model.py lda reduced 5

# --> python ./model/run_model.py ctmp diminished 5
# --> python ./model/run_model.py lda diminished 5

# TODO: when writing settings.txt into "/model" directory, correct some floats into int (e.g, num_topics, user_size...)
# TODO: convert sys.argv into ArgParser

def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ["ctmp", "lda"] or sys.argv[2] not in ["original", "reduced",
                                                                                       "diminished"]:
        print("WRONG USAGE! TRY --> python ./model/run_model.py  [ctmp or lda] [original, reduced or diminished]")
        exit()

    # Get environment variables
    which_model = sys.argv[1]
    which_size = sys.argv[2]
    k_cross_val = int(sys.argv[3])

    docs_file = "./input-data/docs.txt" if which_size == "original" else "./input-data/docs_REDUCED.txt" if which_size == "reduced" else "./input-data/docs_DIMINISHED.txt"
    rating_file = "./input-data/df_rating_UPDATED" if which_size == "original" else "./input-data/df_rating_REDUCED" if which_size == "reduced" else "./input-data/df_rating_DIMINISHED"
    setting_file = "./input-data/settings.txt" if which_size == "original" else "./input-data/settings_REDUCED.txt" if which_size == "reduced" else "./input-data/settings_DIMINISHED.txt"
    output_folder = "./output-data/"

    # Create model folder if it doesn't exist
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # -------------------------------------------- Get Data --------------------------------------------------------
    # Read & write settings into model folder
    print('reading setting ...')
    ddict = utilities.read_setting(setting_file)
    print('write setting ...')
    file_name = f'{output_folder}/setting.txt'
    utilities.write_setting(ddict, file_name)

    """
    wordids: A list whose each element is an array (words ids), corresponding to a document.
             Each element of the array is index of a unique word in the vocabulary.

    wordcts: A list whose each element is an array (word counts), corresponding to a document.
             Each element of the array says how many time the corresponding term in wordids appears
             in the document.

    E.g,
    First document = "Movie is about happy traveler"

    wordids[0] = array([127, 55, 284, 36, 47], dtype=int32)
    first document contains words whose indexes are 127th, 55th, 284th, 36th and 47th in vocabulary

    wordcts[0] = array([1, 1, 1, 1, 1], dtype=int32)
    in first document, words whose indexes are 127, 55, 284, 36, 47 appears 1, 1, 1, 1, 1 times respectively.
     """
    wordids, wordcts = utilities.read_data(docs_file)

    """
    rating_GroupForUser: dictionary where keys are users, values are movies those users liked
    e.g, {48: array([25, 99, 138]), .. } ---> user_id = 48 LIKED movie_id = 25, movie_id = 99, movie_id = 138
    
    rating_GroupForMovie: dictionary where keys are movies, values are users who liked those movies
    e.g, {24: array([13, 55]), .. } ---> movie_id = 24 is LIKED by user_id = 13 and user_id = 55"""

    # Split Ratings into Train/Test with Stratified K-fold Cross-Validation. Save Folds Afterwards.
    # UNCOMMENT below if loading mode is needed
    utilities.cv_train_test_split(rating_file, k_cross_val, seed=42)

    # Load saved Train/Test k-folds
    print(f"LOADING MODE --> Load Train/Test {k_cross_val}-folds ...")
    train_folds = pickle.load(open(f"./input-data/train_{k_cross_val}_folds.pkl", "rb"))
    test_folds = pickle.load(open(f"./input-data/test_{k_cross_val}_folds.pkl", "rb"))

    # Inspect eligibility of folds
    '''for train, test in zip(train_folds, test_folds):
        rating_GroupForUser_train = train[0]
        rating_GroupForUser_test = test[0]

        rating_GroupForMovie_train = train[1]
        rating_GroupForMovie_test = test[1]

        u_test = 0
        for key in rating_GroupForUser_test:
            u_test += len(rating_GroupForUser_test[key])

        u_train = 0
        for key in rating_GroupForUser_train:
            u_train += len(rating_GroupForUser_train[key])

        # Correct is 0.2 --> 5-fold cross validation
        print(u_test / (u_test + u_train))

        m_test = 0
        for key in rating_GroupForMovie_test:
            m_test += len(rating_GroupForMovie_test[key])

        m_train = 0
        for key in rating_GroupForMovie_train:
            m_train += len(rating_GroupForMovie_train[key])

        print(m_test / (m_test + m_train))

        less_test = 0
        for key in rating_GroupForUser_test:
            if len(rating_GroupForUser_test[key]) <= 5:
                less_test += 1

        less_train = 0
        for key in rating_GroupForUser_train:
            if len(rating_GroupForUser_train[key]) <= 5:
                less_train += 1

        # Badly distributed
        print(less_test / len(rating_GroupForUser_test))
        print(less_train / len(rating_GroupForUser_train))

    exit()'''

    for train, test in zip(train_folds, test_folds):
        rating_GroupForUser_train = train[0]
        rating_GroupForUser_test = test[0]

        rating_GroupForMovie_train = train[1]
        rating_GroupForMovie_test = test[1]
        # with open(f"./.test/rating_GroupForUser_train.pkl", "wb") as f:
        #       pickle.dump(rating_GroupForUser_train, f)
        # with open(f"./.test/rating_GroupForMovie_train.pkl", "wb") as f:
        #      pickle.dump(rating_GroupForMovie_train, f)
        # with open(f"./.test/rating_GroupForUser_test.pkl", "wb") as f:
        #      pickle.dump(rating_GroupForUser_test, f)
        # with open(f"./.test/rating_GroupForMovie_test.pkl", "wb") as f:
        #      pickle.dump(rating_GroupForMovie_test, f)
        break

    # -------------------------------------- Initialize Algorithm --------------------------------------------------
    if which_model == "ctmp":
        print('initializing CTMP algorithm ...\n')
        algo = MyCTMP(rating_GroupForUser_train, rating_GroupForMovie_train,
                      ddict['num_docs'], ddict['num_terms'], ddict['num_topics'],
                      ddict["user_size"], ddict["lamb"], ddict["e"], ddict["f"], ddict['alpha'],
                      ddict['iter_infer'])

    else:
        print('initializing LDA algorithm ...\n')
        algo = MyLDA(ddict['num_docs'], ddict['num_terms'], ddict['num_topics'], ddict['alpha'],
                     ddict['iter_infer'])

    # ----------------------------------------- Run Algorithm ------------------------------------------------------
    print('START!')
    for i in range(ddict['iter_train']):
        print(f'\n*** iteration: {i} ***\n')
        time.sleep(2)
        # Run single EM step and return attributes
        algo.run_EM(wordids, wordcts, i)

        # Save CheckPoints
        if i % 5 == 0 and i != 0:
            os.makedirs(f"{output_folder}{i}")
            list_tops = utilities.list_top(algo.beta, ddict['tops'])
            print("\nsaving the final results.. please wait..")
            utilities.write_file(output_folder, list_tops, algo, i)
            # evaluate = MyEvaluation(rating_GroupForUser_train, rating_GroupForUser_test,
            #                        rating_GroupForMovie_train, rating_GroupForMovie_test, i, sample_test=1000)
            # evaluate.plot()

    print('DONE!')

    # ----------------------------------------- Write Results ------------------------------------------------------
    # Search top words of each topics
    # list_tops = utilities.list_top(algo.beta, ddict['tops'])
    # print("\nsaving the final results.. please wait..")
    # utilities.write_file(output_folder, list_tops, algo)


if __name__ == '__main__':
    import os
    NUM_THREADS = "1"
    os.environ["OMP_NUM_THREADS"] = NUM_THREADS
    os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
    os.environ["MKL_NUM_THREADS"] = NUM_THREADS
    os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS
    os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS
    import numpy as np
    import shutil
    import sys
    import time
    import pickle
    import pandas as pd
    from math import floor
    from CTMP import MyCTMP
    # from LDA import MyLDA
    from Evaluation import MyEvaluation

    sys.path.insert(0, './common')
    import utilities

    main()
