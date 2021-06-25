# Bi-LSTM with Multi-features Embedding for Relation Classification

This is a neural network model with multi-features embedding based on Bi-directional Long-Short Term Memory (Bi-LSTM) to classify intra-sentential relation. This model chooses word representation, position features and lexical level features as input features to describe the characteristics at different levels of relation. These features don?¡¥t need complex algorithm calculation, they are suitable for various sentence level relation classification tasks.

The functions of each file are as follows:
config.py is for parameter setting.
LSTM.py in the models folder is the model.
main_sem.py is the running program.
semeval.py in the dataset folder is used to read in corpus data.
The dataset folder stores the data that can be used for experiments. GENIA folder stores the preprocessed GENIA corpus data which can be used for experiments.

