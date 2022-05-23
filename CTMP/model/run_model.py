#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------ RUN in terminal ------------
# --> python ./model/run_model.py ctmp nflx 5 1
# --> python ./model/run_model.py lda nflx 5 1

# --> python ./model/run_model.py ctmp movielens20m 5 1
# --> python ./model/run_model.py lda movielens20m 5 1

# TODO: when writing settings.txt into "/model" directory, correct some floats into int (e.g, num_topics, user_size...)
# TODO: convert sys.argv into ArgParser

def main():
    if len(sys.argv) != 5 or sys.argv[1] not in ["ctmp", "lda"] or sys.argv[2] not in ["nflx", "movielens20m",
                                                                                       "reduced",
                                                                                       "diminished"]:
        print(
            "WRONG USAGE! TRY --> python ./model/run_model.py  [ctmp or lda] [nflx, movielens20m, reduced or diminished]")
        exit()

    # Get environment variables
    which_model = sys.argv[1]
    which_dataset = sys.argv[2]
    k_cross_val = int(sys.argv[3])
    which_k_cross_fold = int(sys.argv[4])

    movies_file = "./input-data/movies_NFLX.txt" if which_dataset == "nflx" else "./input-data/movies.txt" if which_dataset == "movielens20m" else "./input-data/movies_REDUCED.txt" if which_dataset == "reduced" else "./input-data/movies_DIMINISHED.txt"
    ratings_file = "./input-data/df_rating_NFLX_UPDATED" if which_dataset == "nflx" else "./input-data/df_rating_UPDATED" if which_dataset == "movielens20m" else "./input-data/df_rating_REDUCED" if which_dataset == "reduced" else "./input-data/df_rating_DIMINISHED"
    settings_file = "./input-data/settings_NFLX.txt" if which_dataset == "nflx" else "./input-data/settings.txt" if which_dataset == "movielens20m" else "./input-data/settings_REDUCED.txt" if which_dataset == "reduced" else "./input-data/settings_DIMINISHED.txt"
    output_folder = "./output-data"

    # -------------------------------------------- Get Data --------------------------------------------------------
    # Read & write settings into model folder
    print('reading setting ...')
    ddict = utilities.read_setting(settings_file)
    print('write setting ...')
    save_dir = (str(ddict["iter_train"]) + "_" + str(ddict["iter_infer"]) + "_" + str(ddict["num_topics"]) +
                "_" + str(ddict["lamb"]) + "_" + str(ddict["alpha"]) + "_" + str(ddict["bernoulli_p"]))
    if ddict["num_movies"] == 7882:
        save_dir = "NFX_" + save_dir
    else:
        save_dir = "MVL_" + save_dir
        
    os.makedirs(f"{output_folder}/{save_dir}")
    file_name = f'{output_folder}/{save_dir}/settings_NFLX.txt' if which_dataset == "nflx" else f'{output_folder}/{save_dir}/settings.txt'
    utilities.write_setting(ddict, file_name)

    wordids, wordcts = utilities.read_data(movies_file)

    # Split Ratings into Train/Test with Stratified K-fold Cross-Validation. Save Folds Afterwards.
    # UNCOMMENT --> SPLITTING MODE
    # utilities.cv_train_test_split(ratings_file, k_cross_val, seed=42)

    # Load saved Train/Test k-folds
    print(f"LOADING MODE --> Load Train/Test {k_cross_val}-folds ...")
    train_folds = pickle.load(
        open(f"./input-data/train_NFLX_{k_cross_val}_folds.pkl", "rb")) if which_dataset == "nflx" else pickle.load(
        open(f"./input-data/train_{k_cross_val}_folds.pkl", "rb"))
    rating_GroupForUser_train = train_folds[which_k_cross_fold - 1][0]
    rating_GroupForMovie_train = train_folds[which_k_cross_fold - 1][1]

    if which_dataset == "nflx":
        for u in range(479870):
            try:
                rating_GroupForUser_train[u]
            except:
                rating_GroupForUser_train[u] = []

    # -------------------------------------- Initialize Algorithm --------------------------------------------------
    if which_model == "ctmp":
        print('initializing CTMP algorithm ...\n')
        algo = MyCTMP(rating_GroupForUser_train, rating_GroupForMovie_train,
                      ddict['num_movies'], ddict['num_words'], ddict['num_topics'],
                      ddict["user_size"], ddict["lamb"], ddict["e"], ddict["f"], ddict['alpha'],
                      ddict['tau'], ddict['kappa'], ddict['bernoulli_p'], ddict['iter_infer'])

    else:
        # TODO: include tau/kappa into both LDA.py and below
        print('initializing LDA algorithm ...\n')
        algo = MyLDA(ddict['num_movies'], ddict['num_words'], ddict['num_topics'], ddict['alpha'], ddict['iter_infer'])

    # ----------------------------------------- Run Algorithm ------------------------------------------------------
    print('START!')
    for i in range(1, ddict['iter_train'] + 1):
        print(f'\n*** iteration: {i} ***\n')
        time.sleep(2)
        # Run single EM step and return attributes
        algo.run_EM(wordids, wordcts, i)

        if i == ddict['iter_train']:
            list_tops = utilities.list_top(algo.beta, ddict['tops'])
            print("\nsaving the final results.. please wait..")
            utilities.write_file(output_folder, list_tops, algo, save_dir)
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

    import shutil
    import sys
    import time
    import pickle
    from CTMP import MyCTMP
    from LDA import MyLDA
    from Evaluation import MyEvaluation

    sys.path.insert(0, './common')
    import utilities

    main()
