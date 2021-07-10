# Master Thesis -  "Probabilistic models for Recommender Systems"
### Collaborative Topic Model for Poisson distributed ratings (CTMP) with the application of Online Maximum a Posteriori Estimation with Bernoulli randomness (BOPE)      
 
CTMP is a hybrid and interpretable probabilistic content-based collaborative filtering model for recommender system. The model enables both content representation by admixture topic modelling called Latent Dirichlet Allocation (LDA) and computational efficiency from assumption of Poisson distributed ratings, living together under one tightly coupled probabilistic model, thus addressing the limitation of previous methods. The paper was released in April 2018, and it is considered one of the latest approaches in commercial product recommendation (movies, documents, scientific articles).  

BOPE is the inference method used in MAP problems which are non-convex and intractable. It has a fast convergence rate and implicit regularization. The paper was released in May 2020 and it is the latest novel method among MAP estimation methods.   

I have implemented CTMP model augmented with BOPE from scratch in Python and studied its behaviour on MovieLens 20M and NETFLIX datasets regarding the movie recommendations. Experimental studies have been carried out for assessing the ability of model on Recall, Precision, Perplexity, Sparsity, Topic Interpretation and Transfer Learning between datasets.

Technologies: Python(Numpy, Scipy, Numba, Pandas, Matplotlib, NLTK), SQL, Google Cloud 




