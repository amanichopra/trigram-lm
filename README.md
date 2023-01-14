# Trigam Language Modeling

In this project, I use the brown corpus to train a trigram language model. First, I compute the unigram, bigram, and trigram counts. Next, I use the maximum likelihood estimates to compute the raw n-gram probabilities. Third, I compute the smoothed trigram probabilities using linear interpolation. Finally, I train 2 trigram models for a binary text classification task (one model for each label) and select the label corresponding to the model with the lowest perplexity. 

Run ```trigam_model.py``` to run text classification.
