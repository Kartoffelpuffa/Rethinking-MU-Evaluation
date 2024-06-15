# Rethinking Evaluation Methods for Machine Unlearning

## Abstract

Machine *unlearning* refers to methods for deleting information about specific training instances from a trained machine learning model. This enables models to delete user information and comply with privacy regulations.
While retraining the model from scratch on the training set excluding the instances to be *forgotten* would result in a desired unlearned model, owing to the size of datasets and models, it is infeasible.
Hence, unlearning algorithms have been developed, where the goal is to obtain an unlearned model that behaves as closely as possible to the retrained model. 
Consequently, evaluating an unlearning method involves -  (i) randomly selecting a *forget* set (i.e., the training instances to be unlearned), (ii) obtaining an unlearned and a retrained model, and (iii) comparing the performance of the unlearned and the retrained model on the test and forget set. 
However, when the forget set is randomly selected, the unlearned model is almost often similar to the original (i.e., prior to unlearning) model. Hence, it is unclear if the model did really unlearn or simply copied the weights from the original model. For a more robust evaluation, we instead propose to
consider training instances with significant influence on the trained model. When such influential instances are considered in the forget set, we observe that the unlearned model deviates significantly from the retrained model. 
Such deviations are also observed when the size of the forget set is increased. Lastly, choice of dataset for evaluation could also lead to misleading interpretation of results.

## Overview

In this repository we provide the code for our experiments regarding influential and random forget sets on IMDB, SST2 and TREC datasets.
To obtain the data and models for the weight comparisons on the LEDGAR dataset, please refer to the [KGA](https://github.com/Lingzhi-WANG/KGAUnlearn) implementation.

## Requirements

* Python: 3.9+
* Pytorch: 2.2+
* [torch-influence](https://github.com/alstonlo/torch-influence)

## Usage example

Example on how to unlearn 1% of influential data from a DistilBERT model fine-tuned on the SST2 dataset using [KGA](https://github.com/Lingzhi-WANG/KGAUnlearn) unlearning. 

1. Fine-tune DistilBERT on SST2 dataset.
```
python baseline.py --dataset sst2 --epochs 1
```
2. Select influential points using [torch-influence](https://github.com/alstonlo/torch-influence).
```
python influences.py --dataset sst2 --model [original model path]
```
3. Select forget set as 1% of training data acccording to highest influences scores.
```
sst2/data/forget_i_1_3735928559.txt
```
4. Train new, forget and retrained models.
```
python finetune.py --dataset sst2 --sample_ratio 0.05 --epochs 1 --influence --seed 0xDEADBEEF
```
5. Perform unlearning.
```
python kga.py --dataset sst2 --sizes 1 --influence --seed 0xDEADBEEF
```
6. Addtionally evaluate retrained and original model.
```
python retrain.py --dataset sst2 --sizes 1 --influence --seed 0xDEADBEEF
```

