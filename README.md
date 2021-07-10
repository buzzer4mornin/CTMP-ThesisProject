# Master Thesis -  "Probabilistic models for Recommender Systems"
### _Collaborative Topic Model for Poisson distributed ratings (CTMP) with the application of Online Maximum a Posteriori Estimation with Bernoulli randomness (BOPE)._      
 
_**CTMP**_ is a hybrid and interpretable probabilistic content-based collaborative filtering model for recommender system. The model enables both content representation by admixture topic modelling called Latent Dirichlet Allocation (LDA) and computational efficiency from assumption of Poisson distributed ratings, living together under one tightly coupled probabilistic model, thus addressing the limitation of previous methods. The paper was released in April 2018, and it is considered one of the latest approaches in commercial product recommendation (movies, documents, scientific articles).  

_**BOPE**_ is the inference method used in MAP problems which are non-convex and intractable. It has a fast convergence rate and implicit regularization. The paper was released in May 2020 and it is the latest novel method among MAP estimation methods.   

I have implemented CTMP model augmented with BOPE from scratch in Python and studied its behaviour on MovieLens 20M and NETFLIX datasets regarding the movie recommendations. Experimental studies have been carried out for evaluating the ability of model on Recall, Precision, Perplexity, Sparsity, Topic Interpretation and Transfer Learning between datasets. For more details, please refer to [paper in this link.](https://docdro.id/8c4Ze1M)

Technologies: Python(Numpy, Scipy, Numba, Pandas, Matplotlib, NLTK), SQL, Google Cloud 

The most attention is put on Time & Space complexity of the model, thus scientific computing libraries such as Numpy and Scipy are used in nearly all operations along with Numba libray which boosts the computational speed by parallelizing the numpy-heavy functions with JIT(just-in-time compilation). After the model implementation is completed, it is then deployed to Google Cloud's Virtual Machine with high performance CPUs considering that Numpy/Scipy environments are based on BLAST - a high-performance computing architecture for CPU.   

Below, the most important directories are illustrated for the purpose of overview:
```
├── CTMP
│   ├── common
│   ├── experimentation
│   ├── input-data
│   ├── model
│   ├── output-data
├── db-files
│   ├── original-files
│   ├── processed-files
├── pre-CTMP
├── papers
│   ├── others
│   ├── variational-inference

Short Explanation:
CTMP     -> model implementation and experimental studies.
df-files -> data fetch from Oracle Database and first phase of pre-processing.
pre-CTMP -> second phase of pre-processing, i.e. Vocabulary Extraction, Document Representation.
papers   -> papers regarding the models along with the techniques used in the field of recommender systems (e.g, CTPF, CTR, Variational Inference)
 
```
If you want to check the main code where the model is implemented, visit;<br/>
[./CTMP/model/CTMP.py](https://github.com/buzzer4mornin/CTMP-ThesisProject/blob/main/CTMP/model/CTMP.py)<br/>  [./CTMP/model/run_model.py](https://github.com/buzzer4mornin/CTMP-ThesisProject/blob/main/CTMP/model/run_model.py) <br/> [./CTMP/model/Evaluation.py](https://github.com/buzzer4mornin/CTMP-ThesisProject/blob/main/CTMP/model/Evaluation.py)

<br/>

Some results from experimental studies:

- Recall & Precision graph and Sparsity Graph <br/>
<img src="https://raw.githubusercontent.com/buzzer4mornin/CTMP-ThesisProject/main/CTMP/experimentation/recall%26precision/NFLX/p%3D0.7/k%3D50/1/result.png" width="200" height="250" /> <img src="https://raw.githubusercontent.com/buzzer4mornin/CTMP-ThesisProject/main/CTMP/experimentation/sparsity/xx.jpg" width="260" height="250" /> 
