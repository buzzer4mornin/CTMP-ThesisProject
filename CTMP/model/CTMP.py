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
        # Theta is very sparse, so we decide to use smoothing to avoid having extreme sparse theta,
        # therefore increase other proportions a bit
        self.theta = np.load('./input-data/CTMP_initial_theta.npy')
        '''ones_theta = np.argmax(self.theta, axis=1)
        self.theta = np.random.uniform(low=0.005, high=0.015, size=(5, 10))
        for i in range(self.theta.shape[0]):
            self.theta[i][ones_theta[i]] = random.uniform(0.9, 0.95)
        norm = self.theta.sum(axis=1)
        self.theta /= norm[:, np.newaxis]'''

        # Initialize mu (topic offsets)
        self.mu = np.copy(self.theta)  # + np.random.normal(0, self.lamb, self.theta.shape)

        # Initialize phi (rating's variational parameter)
        self.phi = self.get_phi()

        # Initialize shp, rte (user's variational parameters)
        self.shp = np.ones((self.user_size, self.num_topics)) * self.e
        self.rte = np.ones((self.user_size, self.num_topics)) * self.f

    def get_phi(self):
        """ Click to read description

        As we know Φ(phi) has shape of (user_size, num_docs, num_topics)
        which is 3D matrix of shape=(138493, 25900, 50) for ORIGINAL || (1915, 639, 50) for REDUCED

        For ORIGINAL data, it is not possible(memory-wise) to store 3D matrix of shape=(138493, 25900, 50) into single numpy array.
        Therefore, we cut the whole 3D matrix into small chunks of 3D matrix and put them into list and set it as our self.phi
        """

        block_2D = np.zeros(shape=(self.num_docs, self.num_topics))

        # Initiate matrix
        phi_matrices = list()

        # Create small 3D matrices and add them into list
        thousand_block_size = self.user_size // 1000
        phi = np.empty(shape=(1000, self.num_docs, self.num_topics))
        for i in range(1000):
            phi[i, :, :] = block_2D
        for i in range(thousand_block_size):
            phi_matrices.append(phi)

        # Create last remaining 3D matrix and add it into list
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
        # Normalization denominator for mu
        norm_mu = np.copy((self.shp / self.rte).sum(axis=0))

        # UPDATE phi, shp, rte
        for u in range(self.user_size):
            movies_for_u = self.rating_GroupForUser[u]  # list of movie ids liked by user u
            phi_block = self.phi[u // 1000]             # access needed 3D matrix of phi list by index
            usr = u % 1000                              # convert user id into interval 0-1000

            # if user didnt like any movie, then dont update anything, continue!
            if len(movies_for_u) == 0:
                continue

            # compute Φuj then normalize it
            phi_uj = np.exp(np.log(self.mu[[movies_for_u], :]) + special.psi(self.shp[u, :]) - np.log(self.rte[u, :]))
            phi_uj_sum = np.copy(phi_uj)[0].sum(axis=1)
            phi_uj_norm = np.copy(phi_uj) / phi_uj_sum[:, np.newaxis]
            # update user's phi in phi_block with newly computed phi_uj_sum
            phi_block[usr, [movies_for_u], :] = phi_uj_norm

            # update user's shp and rte
            self.shp[u, :] = self.e + phi_uj_norm[0].sum(axis=0)
            self.rte[u, :] = self.f + self.mu.sum(axis=0)
            print(f" ** UPDATE phi, shp, rte over {u + 1}/{self.user_size} users |iter:{self.GLOB_ITER}| ** ")

        # UPDATE theta, mu
        # norm_mu = np.copy((self.shp / self.rte).sum(axis=0))
        for d in range(self.num_docs):
            thetad = self.update_theta(wordids[d], wordcts[d], d)
            self.theta[d, :] = thetad

            mud = self.update_mu(norm_mu, d)
            self.mu[d, :] = mud

            print(f" ** UPDATE theta, mu over {d + 1}/{self.num_docs} documents |iter:{self.GLOB_ITER}| ** ")

    def update_mu(self, norm_mu, d):
        # initiate new mu
        mu = np.empty(self.num_topics)
        mu_users = self.rating_GroupForMovie[d]

        def get_phi(x):
            phi_ = self.phi[x // 1000]
            usr = x % 1000
            return phi_[usr, d, :]  # **previously** np.sum()

        rating_phi = sum(map(get_phi, mu_users))

        if len(mu_users) == 0:
            mu = np.copy(self.theta[d, :])
        else:
            for k in range(self.num_topics):
                temp = -1 * norm_mu[k] + self.lamb * self.theta[d, k]
                delta = temp ** 2 + 4 * self.lamb * rating_phi[k]  # added [k] to rating_phi.
                mu[k] = (temp + np.sqrt(delta)) / (2 * self.lamb)
            # for k in range(self.num_topics):
            #    mu[k] = rating_phi[k] / norm_mu[k]
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
