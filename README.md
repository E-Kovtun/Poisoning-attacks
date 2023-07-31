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

   
