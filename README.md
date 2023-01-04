# Trigam Language Modeling

In this project, I use the brown corpus to train a trigram language model. First, I compute the unigram, bigram, and trigram counts. Next, I use the maximum likelihood estimates to compute the raw n-gram probabilities. Thirs, I compute the smoothed trigram probabilities using linear interpolation. Then, we evaluate the model intrinsically using perplexity on the training corpus. Lastly, I train 2 trigram models for a binary text classification task (one model for each label) and select the label corresponding to the model with the lowest perplexity.
