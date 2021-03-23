# -*- coding: utf-8 -*-
import numpy as np


class MyLDA:
    def __init__(self, num_docs, num_words, num_topics, alpha, iter_infer):
        """ Click to read more

        Arguments:
            num_docs: Number of documents
            num_words: Number of unique words in the corpus (length of the vocabulary).
            num_topics: Number of topics shared by the whole corpus.
            alpha: Hyperparameter for prior on topic mixture theta.
            iter_infer: Number of iterations for BOPE algorithm
          """
        self.num_docs = num_docs
        self.num_words = num_words
        self.num_topics = num_topics
        self.alpha = alpha
        self.iter_infer = iter_infer

        # Initialize beta (topics)
        self.beta = np.random.rand(self.num_topics, self.num_words) + 1e-10
        beta_norm = self.beta.sum(axis=1)
        self.beta /= beta_norm[:, np.newaxis]

        # Initialize theta (topic proportions)
        self.theta = np.random.rand(self.num_docs, self.num_topics) + 1e-10
        theta_norm = self.theta.sum(axis=1)
        self.theta /= theta_norm[:, np.newaxis]

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
        """ Does E step: update theta for all documents """
        for d in range(self.num_docs):
            thetad = self.update_theta(wordids[d], wordcts[d], d)
            self.theta[d, :] = thetad
            print(f" ** UPDATE theta over {d+1}/{self.num_docs} documents |iter:{self.GLOB_ITER}| ** ")

    def update_theta(self, ids, cts, d):
        """ Click to read more

        Updates theta for a given document using BOPE algorithm (i.e, MAP Estimation With Bernoulli Randomness).

        Arguments:
        ids: an element of wordids, corresponding to a document.
        cts: an element of wordcts, corresponding to a document.

        Returns updated theta.
        """

        # Get beta
        beta = self.beta[:, ids]

        # Get theta
        theta = self.theta[d, :]

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

            G_1 = np.dot(beta, cts / x) / p
            G_2 = (self.alpha - 1) / theta / (1 - p)

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
            x_l = np.dot(cts, np.log(np.dot(theta_lower, beta))) + (self.alpha - 1) * np.log(theta_lower)
            x_u = np.dot(cts, np.log(np.dot(theta_upper, beta))) + (self.alpha - 1) * np.log(theta_upper)

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
        """ Does M step: update global variables beta """

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
