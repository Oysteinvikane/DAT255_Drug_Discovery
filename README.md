# DAT255_Drug_Discovery
This project is a part of the course DAT255 Deep Learning at Western Norway University of applied science.

## Introduction
The development of new drugs takes between 15-20 years of development and costs an average of 4 billion US dollars. Modern drug discovery and development usually follows a set of systematic procedures. Modern drug discovery and drug development usually follows a set of systematic procedures, often starting with a validated biomolecular target for which one tries to find a modulator (i.e. a drug). For many years machine learning has played a role in drug discovery. In recent years modern machine learning techniques have been applied to all stages, from initial target discovery and compound screening to the analysis of pathology data during clinical development.

![](images/Drugdevmodel.jpg)

In this project we will focus on early stages of this development, more specifically predicting a candidates affinity and activity against a target. This is a long and tedious search to find a drug-like small molecule called a development candidate. The candidate is then choosen for the clinical development stage. There are strong implications that this process can be sped up with machine learning techniques.

We will work closely and to a degree try to recreate this article:

Koutsoukas, A., Monaghan, K.J., Li, X. et al. Deep-learning: investigating deep neural networks hyper-parameters and comparison of performance to shallow methods for modeling bioactivity data. J Cheminform 9, 42 (2017). https://doi.org/10.1186/s13321-017-0226-y


Blogpost for the project can be found at: https://drug-discovery.github.io/Drug-Discovery-blog/

## Project organization

    ├── README.md          <- The top-level README.
    ├── dataset
    │   ├── 13321_2017_226_MOESM1_ESM   <- Data from third party sources.
    │   ├── temp                        <- Temporary data that has been transformed.
    │
    ├── Notebook          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         e.g. "1.0-data_and_RF-model"
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── images        <- figures to be used in README.md
    │
    ├── environment.yml   <- The conda and pip requirements file for reproducing the analysis environment
