# Poisoning-attacks

This repository is devoted to applying poisoning attacks on deep models for transaction data. 
Four datasets are considered:
1) Churn
2) Age
3) Raif
4) Default

We solve binary classification problem for these datasets. Three of them (Churn, Age, and Raif) are balanced, while Default dataset is imbalanced one. Our goal is to classify an input sample that is represented in form of sequence of transactions (mcc code of transaction). 

We implement four models for the classification task:
1) LSTM
2) LSTM with attention
3) CNN
4) Transformer (only encoder)

Pipeline of dataset preprocessing can be found at `./DataPreprocessing/`. Model implementation is at `./models/`.

### Clean models 

To train clean models (on initial data) run:
```
bash run.sh
```

Model checkpoints will be saved at folder `./checkpoints/clean`. Metrics calculated on the basis of 5 launches will be saved at folder `./results/clean`. 

### Poisoning attacks
In case of binary classification task, the goal of poisoning is to add a specific token during inderence to the transaction sequnces to make the trained model misclassify the sample (predict 0 instead of 1 and vice versa)

There are two attack strategies:
1) 'new_ptokens': We initialize two poisoning tokens (one for each class) that are not represented in initial sequences (out of vocabulary). We add these tokens to the end f sequences to poison them.
2) 'composed_ptokens': We construct poisoning structures from vocabulary. For example, we can take a pair of random tokens and poison examples with it.

To poison the model, run:
```
bash exp.sh
```
Model checkpoints will be saved at folder `./checkpoints/poison`. Metrics calculated on the basis of 5 launches will be saved at folder `./results/poison`. 



