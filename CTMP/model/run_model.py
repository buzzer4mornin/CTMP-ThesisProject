#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import shutil
import sys
import time
import numpy as np
from CTMP import MyCTMP
from LDA import MyLDA

sys.path.insert(0, './common')
import utilities


# RUN --> python ./model/run_model.py ctmp original
# RUN --> python ./model/run_model.py lda original

# RUN --> python ./model/run_model.py ctmp reduced
# RUN --> python ./model/run_model.py lda reduced

# RUN --> python ./model/run_model.py ctmp diminished
# RUN --> python ./model/run_model.py lda diminished

# TODO: when writing settings.txt into "/model" directory, correct some floats into int (e.g, num_topics, user_size...)

def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ["ctmp", "lda"] or sys.argv[2] not in ["original", "reduced", "diminished"]:
        print("WRONG USAGE! TRY --> python ./model/run_model.py  [ctmp or lda] [original, reduced or diminished]")
        exit()

    # Get environment variables
    which_model = sys.argv[1]
    which_size = sys.argv[2]

    docs_file = "./input-data/docs.txt" if which_size == "original" else "./input-data/docs_REDUCED.txt" if which_size == "reduced" else "./input-data/docs_DIMINISHED.txt"
    rating_file = "./input-data/df_rating" if which_size == "original" else "./input-data/df_rating_REDUCED" if which_size == "reduced" else "./input-data/df_rating_DIMINISHED"
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
    rating_GroupForUser, rating_GroupForMovie = utilities.get_rating_group(rating_file)
    # for key in rating_GroupForMovie:
    #    if len(rating_GroupForMovie[key]) > 10000:
    #        print(rating_GroupForMovie[key])
    #exit()
    # -------------------------------------- Initialize Algorithm --------------------------------------------------
    if which_model == "ctmp":
        print('initializing CTMP algorithm ...\n')
        algo = MyCTMP(rating_GroupForUser, rating_GroupForMovie,
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
        time.sleep(4)
        # run single EM step and return attributes
        algo.run_EM(wordids, wordcts, i)

    print('DONE!')

    # ----------------------------------------- Write Results ------------------------------------------------------
    # Search top words of each topics
    list_tops = utilities.list_top(algo.beta, ddict['tops'])

    print("\nsaving the final results.. please wait..")
    utilities.write_file(output_folder, list_tops, algo)


if __name__ == '__main__':
    main()
