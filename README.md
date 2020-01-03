# Learning Connectivity Structure for Brain Signal Classification
This repository contains the official PyTorch implementation of:

**Brain Signal Classification via Learning Connectivity Structure**  
Soobeom Jang, Seong-Eun Moon, Jong-Seok Lee

https://arxiv.org/abs/1905.11678

### Abstract
Connectivity between different brain regions is one of the most important properties for classification of brain signals including electroencephalography (EEG). However, how to define the connectivity structure for a given task is still an open problem, because there is no ground truth about how the connectivity structure should be in order to maximize the classification performance. In this paper, we propose an end-to-end neural network model for EEG classification, which can extract an appropriate multi-layer graph structure and signal features directly from a set of raw EEG signals and perform classification. Experimental results demonstrate that our method yields improved performance in comparison to the existing approaches where manually defined connectivity structures and signal features are used. Furthermore, we show that the graph structure extraction process is reliable in terms of consistency, and the learned graph structures make much sense in the neuroscientific viewpoint.

### Data generation
Data generation code is attached at data/lmdb_generate.py, which converts DEAP(https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) and DREAMER(https://zenodo.org/record/546113) dataset to LMDB format. 
The processed DEAP dataset needs ~12GB, and DREAMER dataset needs ~2GB.
To run the generation code, pip lmdb package is required.

### Run experiments
python train.py

### Acknowledgement
We referenced the implementation of [Neural Relational Inference](https://github.com/ethanfetaya/NRI), which is the official PyTorch version of the paper [Neural Relational Inference for Interacting Systems](https://arxiv.org/abs/1802.04687).


