# -*- coding: utf-8 -*-

import time
import numpy as np
from sys import getsizeof
from scipy import special
import random


class MyCTMP:
    def __init__(self, rating_GroupForUser, rating_GroupForMovie,
                 num_docs, num_words, num_topics, user_size, lamb, e, f, alpha, iter_infer):
        """
        Arguments:
            num_words: Number of unique words in the corpus (length of the vocabulary).
            num_topics: Number of topics shared by the whole corpus.
            alpha: Hyperparameter for prior on topic mixture theta.
            iter_infer: Number of iterations of FW algorithm
          """
        self.rating_GroupForUser = rating_GroupForUser
        self.rating_GroupForMovie = rating_GroupForMovie
        self.num_docs = num_docs
        self.num_words = num_words
        self.num_topics = num_topics
        self.user_size = user_size
        self.lamb = lamb
        self.e = e
        self.f = f
        self.alpha = alpha
        self.iter_infer = iter_infer

        # Get initial beta(topics) which was produced by LDA
        self.beta = np.load('./input-data/CTMP_initial_beta.npy')

        # Get initial theta(topic proportions) which was produced by LDA
        self.theta = np.load('./input-data/CTMP_initial_theta.npy')

        # Initialize mu (topic offsets)
        self.mu = np.copy(self.theta)  # + np.random.normal(0, self.lamb, self.theta.shape)

        # Initialize phi (rating's variational parameter)
        self.phi = self.get_phi()

        # Initialize shp, rte (user's variational parameters)
        self.shp = np.ones((self.user_size, self.num_topics)) * self.e
        self.rte = np.ones((self.user_size, self.num_topics)) * self.f
        # INIT shp, rte
        # self.shp, self.rte = self.get_shp_rte()
        # self.shp = np.load("./input-data/CTMP_initial_shp.npy")
        # self.rte = np.load("./input-data/CTMP_initial_rte.npy")

    '''def get_shp_rte(self):
        # init
        shp = np.empty(shape=(self.user_size, self.num_topics))
        rte = np.empty(shape=(self.user_size, self.num_topics))

        for u in range(self.user_size):
            movies_for_u = self.rating_GroupForUser[u]
            phi_block = self.phi[u // 1000]
            usr = u % 1000

            if len(movies_for_u) == 0:
                shp[u, :] = 0.3  # float(self.e)
                rte[u, :] = self.f + self.mu.sum(axis=0)
                continue

            temp = np.copy(phi_block[usr, [movies_for_u], :])
            shp[u, :] = self.e + np.sum(temp[0], axis=0)
            rte[u, :] = self.f + self.mu.sum(axis=0)

            print(f" ** INIT shp, rte over {u + 1}/{self.user_size} users ** ")
        return shp, rte'''

    def get_phi(self):
        """ Click to read description

        As we know Φ(phi) has shape of (user_size X num_docs X num_topics)
        which is 3D matrix of shape=(138493, 25900, 50) for ORIGINAL || (1915, 639, 50) for REDUCED

        For ORIGINAL data, it is not possible(memory-wise) to have a numpy array of shape=(138493, 25900, 50)
        Therefore, we cut the whole 3D matrix into small chunks of 3D matrix and put them into list - phi_matrices()
        """

        # Randomly generate 2D matrix block
        # block_2D = np.random.rand(self.num_docs, self.num_topics) + 1e-10
        # block_2D_norm = block_2D.sum(axis=1)
        # block_2D /= block_2D_norm[:, np.newaxis]
        block_2D = np.zeros(shape=(self.num_docs, self.num_topics))

        # Initiate matrix
        phi_matrices = list()

        # Create small chunks of 3D matrix and add them into list
        # Phi of user_ids in interval - [0, 137999] for ORIGINAL
        # Phi of user_ids in interval - [0, 999] for REDUCED
        thousand_block_size = self.user_size // 1000
        phi = np.empty(shape=(1000, self.num_docs, self.num_topics))
        for i in range(1000):
            phi[i, :, :] = block_2D
        for i in range(thousand_block_size):
            phi_matrices.append(phi)

        # Create remaining chunk of 3D matrix and add it into list
        # Phi of user_ids in interval - [138000, 138493] for ORIGINAL
        # Phi of user_ids in interval - [1000, 1915] for REDUCED
        remaining_block_size = self.user_size % 1000
        phi = np.empty(shape=(remaining_block_size, self.num_docs, self.num_topics))
        for i in range(remaining_block_size):
            phi[i, :, :] = block_2D
        phi_matrices.append(phi)

        return phi_matrices

    def run_EM(self, wordids, wordcts, GLOB_ITER):
        """ Click to read more

        First does an E step on given wordids and wordcts to update theta,
        then uses the result to update betas in M step.
        """
        self.GLOB_ITER = GLOB_ITER

        # E - expectation step
        self.e_step(wordids, wordcts)
        # M - maximization step
        self.m_step(wordids, wordcts)

    def e_step(self, wordids, wordcts):
        """ Does e step. Updates theta, mu, pfi, shp, rte for all documents and users"""
        # UPDATE phi, shp, rte
        for u in range(self.user_size):
            movies_for_u = self.rating_GroupForUser[u]  # get movies liked by user u
            phi_block = self.phi[u // 1000]  # access the needed small chunk of phi by list index
            usr = u % 1000  # get user for that small chunk of phi

            # if user didnt like any movie CHECK CORRECTNESS
            if len(movies_for_u) == 0:
                self.shp[u, :] = 0.3  # float(self.e)
                self.rte[u, :] = self.f + self.mu.sum(axis=0)
                print(f" **** UPDATE phi, shp, rte over {u + 1}/{self.user_size} users |iter:{self.GLOB_ITER}| ** ")
                continue

            temp = np.exp(np.log(self.mu[[movies_for_u], :]) + special.psi(self.shp[u, :]) - np.log(self.rte[u, :]))
            temp_sum = np.copy(temp)[0].sum(axis=1)
            temp_norm = np.copy(temp) / temp_sum[:, np.newaxis]
            phi_block[usr, [movies_for_u], :] = temp_norm
            self.shp[u, :] = self.e + temp_norm[0].sum(axis=0)
            self.rte[u, :] = self.f + self.mu.sum(axis=0)
            print(f" ** UPDATE phi, shp, rte over {u + 1}/{self.user_size} users |iter:{self.GLOB_ITER}| ** ")

        # UPDATE theta, mu
        div = (self.shp / self.rte).sum(axis=0)
        for d in range(self.num_docs):
            thetad = self.update_theta(wordids[d], wordcts[d], d)
            self.theta[d, :] = thetad

            mud = self.update_mu(div, d)
            self.mu[d, :] = mud
            print(f" ** UPDATE theta, mu over {d + 1}/{self.num_docs} documents |iter:{self.GLOB_ITER}| ** ")

    def update_mu(self, div, d):
        # initiate new mu
        mu = np.empty(self.num_topics)
        mu_users = self.rating_GroupForMovie[d]

        def _get_rating_phi(x):
            phi_ = self.phi[x // 1000]
            usr = x % 1000
            return np.sum(phi_[usr, d, :])

        rating_phi = sum(map(_get_rating_phi, mu_users))
        # rating_phi = float(len(mu_users))  # ALTERNATIVELY! but check if it is must -> sum(phi)==1

        if rating_phi == 0:
            # TODO: is this approach enough?
            mu = np.copy(self.theta[d, :])
        else:
            # for k in range(self.num_topics):
            #    temp = -1 * div[k] + self.lamb * self.theta[d, k]
            #    delta = temp ** 2 + 4 * self.lamb * rating_phi
            #    mu[k] = (temp + np.sqrt(delta)) / (2 * self.lamb)
            #    if mu[k] > 100:
            #        print(rating_phi)
            #        print(temp)
            #        print(delta)
            #        exit()
            for k in range(self.num_topics):
                mu[k] = rating_phi / div[k]
                #if mu[k] > 100:
                #    print(mu[k])
                #    print(rating_phi)
                #    print(div[k])
                #    exit()
        return mu

    def update_theta(self, ids, cts, d):
        """ Click to read more

        Updates theta for a given document using BOPE algorithm (i.e, MAP Estimation With Bernoulli Randomness).

        Arguments:
        ids: an element of wordids, corresponding to a document.
        cts: an element of wordcts, corresponding to a document.

        Returns updated theta.
        """

        # locate cache memory
        beta = self.beta[:, ids]

        # Get theta
        theta = self.theta[d, :]

        # Get mu
        mu = self.mu[d, :]

        # x = sum_(k=2)^K theta_k * beta_{kj}
        x = np.dot(theta, beta)

        # Parameter of Bernoulli distribution
        # Likelihood vs Prior
        p = 0.5

        # Number of times likelihood and prior are chosen
        T_lower = [1, 0]
        T_upper = [0, 1]

        for t in range(1, self.iter_infer):
            # ======== Lower ==========
            if np.random.rand() < p:
                T_lower[0] += 1
            else:
                T_lower[1] += 1

            G_1 = (np.dot(beta, cts / x) + (self.alpha - 1) / theta) / p
            G_2 = (-1 * self.lamb * (theta - mu)) / (1 - p)

            ft_lower = T_lower[0] * G_1 + T_lower[1] * G_2
            index_lower = np.argmax(ft_lower)
            alpha = 1.0 / (t + 1)
            theta_lower = np.copy(theta)
            theta_lower *= 1 - alpha
            theta_lower[index_lower] += alpha

            # ======== Upper ==========
            if np.random.rand() < p:
                T_upper[0] += 1
            else:
                T_upper[1] += 1

            ft_upper = T_upper[0] * G_1 + T_upper[1] * G_2
            index_upper = np.argmax(ft_upper)
            alpha = 1.0 / (t + 1)
            theta_upper = np.copy(theta)
            theta_upper *= 1 - alpha
            theta_upper[index_upper] += alpha
            # print(theta_upper - theta_lower)

            # ======== Decision ========
            x_l = np.dot(cts, np.log(np.dot(theta_lower, beta))) + (self.alpha - 1) * np.log(theta_lower) \
                  - 1 * (self.lamb / 2) * (np.linalg.norm((theta_lower - mu), ord=2)) ** 2
            x_u = np.dot(cts, np.log(np.dot(theta_upper, beta))) + (self.alpha - 1) * np.log(theta_upper) \
                  - 1 * (self.lamb / 2) * (np.linalg.norm((theta_upper - mu), ord=2)) ** 2

            compare = np.array([x_l[0], x_u[0]])
            best = np.argmax(compare)

            # ======== Update ========
            if best == 0:
                theta = np.copy(theta_lower)
                x = x + alpha * (beta[index_lower, :] - x)
            else:
                theta = np.copy(theta_upper)
                x = x + alpha * (beta[index_upper, :] - x)
        return theta

    def m_step(self, wordids, wordcts):
        """ Does m step: update global variables beta """

        # Compute intermediate beta which is denoted as "unit beta"
        beta = np.zeros((self.num_topics, self.num_words), dtype=float)
        for d in range(self.num_docs):
            beta[:, wordids[d]] += np.outer(self.theta[d], wordcts[d])
        # Check zeros index
        beta_sum = beta.sum(axis=0)
        ids = np.where(beta_sum != 0)[0]
        unit_beta = beta[:, ids]
        # Normalize the intermediate beta
        unit_beta_norm = unit_beta.sum(axis=1)
        unit_beta /= unit_beta_norm[:, np.newaxis]
        # Update beta
        self.beta = np.zeros((self.num_topics, self.num_words), dtype=float)
        self.beta[:, ids] += unit_beta
