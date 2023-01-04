import sys
from collections import defaultdict, Counter
import math
import random
import os
import os.path
import warnings
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
"""
COMS W4705 - Natural Language Processing - Fall 2022 
Prorgramming Homework 1 - Trigram Language Models
Daniel Bauer
"""


def corpus_reader(corpusfile, lexicon=None, read_limit=None):
    with open(corpusfile,'r') as corpus: 
        for i, line in enumerate(corpus):
            if read_limit and i >= read_limit: break
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence


def get_lexicon(corpus, read_limit=None):
    word_counts = defaultdict(int)
    for i, sentence in enumerate(corpus):
        if read_limit and i >= read_limit: break
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  


def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    if len(sequence) == 0:
        return []

    if n == 1:
        sequence = ['START'] + sequence
    sequence = ['START' for i in range(n-1)] + sequence + ['STOP']
    return [tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]


# These tests are inspired by a github I found https://github.com/geraldzakwan/nlp-fall-2019/blob/master/hw1/trigram_model.py.
# I changed much of the tests to reflect my understanding and simply copied the format of the tests.
def run_tests(data_dir='./data', corpusfile_read_limit=None):
    # build model

    model = TrigramModel(f'{data_dir}/brown_train.txt', corpusfile_read_limit=corpusfile_read_limit)
    # get_ngrams()
    assert get_ngrams(['natural', 'language', 'processing'], 1) == [("START",),
                                                                    ('natural',),
                                                                    ('language',),
                                                                    ('processing',),
                                                                    ('STOP',)]
    assert get_ngrams(['natural', 'language', 'processing'], 2) == [("START", 'natural',),
                                                                    ('natural', 'language'),
                                                                    ('language', 'processing'),
                                                                    ('processing', "STOP")]
    assert get_ngrams(['natural', 'language', 'processing'], 3) == [("START", "START", 'natural'),
                                                                    ("START", 'natural', 'language'),
                                                                    ('natural', 'language', 'processing'),
                                                                    ('language', 'processing', 'STOP')]

    # count_ngrams()
    if not corpusfile_read_limit:
        assert model.unigramcounts[('START',)] == 41614
        assert model.unigramcounts[('START',)] == model.unigramcounts[('STOP',)]
        assert model.unigramcounts[('the',)] == 61428
        assert model.bigramcounts[('START', 'the')] == 5478
        assert model.trigramcounts[('START', 'START', 'the')] == 5478

        assert model.training_corpus_num_words == (1084179 - 41614) # subtract # start words

        # raw_uni/bi/trigram_probability()
        assert model.raw_unigram_probability(('the',)) == 61428 / (1084179 - 41614)
        assert model.raw_bigram_probability(('the', 'jury')) == 35 / 61428
        assert model.raw_trigram_probability(('the', 'jury', 'said')) == 7 / 35

    # zero probabilities
    assert model.raw_unigram_probability(('ooshitbot',)) == 0  # no such unigram
    assert model.raw_bigram_probability(('the', 'ooshtibot')) == 0  # no such bigram (numerator is zero)
    assert model.raw_bigram_probability(
        ('ooshtibot', 'the')) == model.raw_unigram_probability(('the',))  # no such preceding unigram (denominator is zero), return probability of word without context
    assert model.raw_trigram_probability(('the', 'jury', 'ooshtibot')) == 0  # no such trigram (numerator is zero)
    assert model.raw_trigram_probability(
        ('ooshtibot', 'the', 'jury')) == model.raw_unigram_probability(('jury',))  # no such preceding bigram (denominator is zero), return probability of word without context

    # special cases for ('START', 'START', anything)
    if not corpusfile_read_limit:
        assert model.raw_bigram_probability(('START', 'the')) == 5478 / 41614
    assert model.raw_trigram_probability(('START', 'START', 'the')) == model.raw_bigram_probability(('START', 'the'))

    # ensure raw P(w_i) sums to 1  for every possible word
    assert abs(sum(model.raw_unigram_probability(unigram) for unigram in model.unigramcounts.keys()) - 1) < 0.01

    # ensure raw P(w_i | w1) sums to 1 for every possible word
    sum_probs = {unigram: 0 for unigram in model.unigramcounts.keys() if unigram[0] != 'STOP'}
    for bigram in model.bigramcounts.keys():
        sum_probs[tuple(bigram[:1])] += model.raw_bigram_probability(bigram)
    for unigram in sum_probs:
        assert abs(sum_probs[unigram] - 1) < 0.01

    # ensure raw P(w_i | w1, w2) sums to 1 for every possible word
    sum_probs = {bigram: 0 for bigram in model.bigramcounts.keys() if bigram[1] != 'STOP'}
    sum_probs[tuple(['START', 'START'])] = 0 # special case
    for trigram in model.trigramcounts.keys():
        sum_probs[tuple(trigram[:2])] += model.raw_trigram_probability(trigram)
    for bigram in sum_probs:
        assert abs(sum_probs[bigram] - 1) < 0.01

    # smoothed_trigram_probability()
    if not corpusfile_read_limit:
        assert model.smoothed_trigram_probability(('ooshtibot', 'the', 'jury')) == (1 / 3) * model.raw_unigram_probability(('jury',)) + (1 / 3) * (
                    35 / 61428) + (1 / 3) * (59 / (1084179 - 41614))  # trigram is not found, but bigram and unigram are found
        assert model.smoothed_trigram_probability(('the', 'ooshtibot', 'jury')) == (1 / 3) * model.raw_unigram_probability(('jury',)) + (1 / 3) * model.raw_unigram_probability(('jury',)) + (
                    1 / 3) * (59 / (1084179 - 41614))  # only unigram is found
    assert model.smoothed_trigram_probability(
        ('the', 'jury', 'ooshtibot')) == 0  # All trigram, bigram and unigram are not found

    # ensure smoothed P(w_i | w1, w2) sums to 1 for every possible word
    sum_probs = {bigram: 0 for bigram in model.bigramcounts.keys() if bigram[1] != 'STOP'}
    sum_probs[tuple(['START', 'START'])] = 0  # special case
    for unigram in tqdm(model.unigramcounts.keys()):
        if unigram[0] == 'START': continue
        for bigram in sum_probs:
            trigram = tuple(list(bigram) + [unigram[0]])
            sum_probs[bigram] += model.smoothed_trigram_probability(trigram)
    for bigram in sum_probs:
        assert abs(sum_probs[bigram] - 1) < 0.01

    # sentence_logprob() The trial had packed the large courtroom for more than a week .
    assert model.sentence_logprob(
        ['his', 'petition', 'charged']) == sum([
        math.log2(model.smoothed_trigram_probability(('START', 'START', 'his'))),
        math.log2(model.smoothed_trigram_probability(('START', 'his', 'petition'))),
        math.log2(model.smoothed_trigram_probability(('his', 'petition', 'charged'))),
        math.log2(model.smoothed_trigram_probability(('petition', 'charged', 'STOP')))
    ])

    # perplexity()
    dev_corpus = corpus_reader(f'{data_dir}/brown_test.txt', model.lexicon)
    pp = model.perplexity(dev_corpus)
    assert pp < 400.0

    # classification accuracy
    acc = essay_scoring_experiment(f'{data_dir}/ets_toefl_data/train_high.txt', f'{data_dir}/ets_toefl_data/train_low.txt', f'{data_dir}/ets_toefl_data/test_high', f'{data_dir}/ets_toefl_data/test_low')
    assert acc > 0.8


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):
    model1 = TrigramModel(training_file1)  # model trained with high scores
    model2 = TrigramModel(training_file2)  # model trained with low scores

    total = 0
    correct = 0

    for f in os.listdir(testdir1):  # high test data
        p1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
        p2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
        if p1 < p2:
            correct += 1
        total += 1

    for f in os.listdir(testdir2):  # low test data
        p1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
        p2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
        if p2 < p1:
            correct += 1
        total += 1
    return correct / total


class TrigramModel(object):
    def __init__(self, corpusfile, corpusfile_read_limit=None):
        self.training_corpusfile = corpusfile

        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile, read_limit=corpusfile_read_limit)
        self.lexicon = get_lexicon(generator, read_limit=corpusfile_read_limit)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")

        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, lexicon=self.lexicon, read_limit=corpusfile_read_limit)
        self.count_ngrams(generator)

        # Count num words in corpus
        self.training_corpus_num_words = sum([self.unigramcounts[unigram] for unigram in self.unigramcounts if unigram != ('START',)])
        if self.training_corpus_num_words == 0:
            warnings.warn("Number of words in corpus is 0! This language model won't model anything!")

    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
        unigrams = []
        bigrams = []
        trigrams = []
        for sentence in corpus:
            unigrams += get_ngrams(sentence, 1)
            bigrams += get_ngrams(sentence, 2)
            trigrams += get_ngrams(sentence, 3)

        self.unigramcounts = Counter(unigrams)
        self.bigramcounts = Counter(bigrams)
        self.trigramcounts = Counter(trigrams)

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        # In general, P(trigram[2] | (trigram[0], trigram[1])) = count(trigram) / count((trigram[0], trigram[1]))

        # Special case for ('START', 'START', word):
        # P(word | ('START', 'START')) = P(word | ('START'))
        if trigram[0] == 'START' and trigram[1] == 'START':
            return self.raw_bigram_probability(tuple(['START', trigram[2]]))
        else:
            denominator = self.bigramcounts[tuple(trigram[:2])]

        if denominator == 0:
            warnings.warn(f'Probability of trigram {trigram} is not well defined! Returning probability of {trigram[2]} without context.')
            return self.raw_unigram_probability(tuple([trigram[2]]))

        return self.trigramcounts[trigram] / denominator

    def get_training_corpus_total_words(self):
        return self.training_corpus_num_words

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        denominator = self.unigramcounts[tuple([bigram[0]])]

        if denominator == 0:
            warnings.warn(f'Probability of bigram {bigram} is not well defined! Returning probability of {bigram[1]} without context.')
            return self.raw_unigram_probability(tuple([bigram[1]]))

        return self.bigramcounts[bigram] / denominator
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.
        if unigram == ('START',):
            return 0.0
        return self.unigramcounts[unigram] / self.training_corpus_num_words

    def generate_sentence(self,t=20, token_selection_method='random'):
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        if token_selection_method not in ['random', 'optimal']:
            raise Exception("Invalid method! Options are ['random', 'optimal'].")

        sentence = []
        previous_tokens = ('START', 'START')

        while len(sentence) < t:
            next_token_options = [trigram[-1] for trigram in self.trigramcounts if
                                  tuple(trigram[:2]) == previous_tokens]
            next_token_probs = [self.raw_trigram_probability(tuple(list(previous_tokens) + [next_token])) for next_token
                                in next_token_options]
            #print({token:prob for token, prob in zip(next_token_options, next_token_probs)})
            if token_selection_method == 'random':
                next_token_choice = random.choices(next_token_options, weights=next_token_probs, k=1)[0]
            elif token_selection_method == 'optimal':
                next_token_choice = next_token_options[np.argmax(next_token_probs)]
            sentence.append(next_token_choice)
            if next_token_choice == 'STOP':
                return ' '.join(sentence)
            previous_tokens = tuple([previous_tokens[-1]] + [next_token_choice])

        return ' '.join(sentence + ['STOP'])

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        # print(self.raw_trigram_probability(trigram))
        # print(self.raw_bigram_probability(tuple([trigram[1], trigram[2]])))
        # print(self.raw_unigram_probability(tuple([trigram[2]])))
        return lambda1*self.raw_trigram_probability(trigram) + lambda2*self.raw_bigram_probability(tuple([trigram[1], trigram[2]])) + \
               lambda3*self.raw_unigram_probability(tuple([trigram[2]]))
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigrams = get_ngrams(sentence, 3)
        sentence_prob = 0
        for trigram in trigrams:
            smoothed_trigram_prob = self.smoothed_trigram_probability(trigram)
            if smoothed_trigram_prob == 0:
                warnings.warn(f'For trigram: {trigram}, either the context or unigram has not been seen in the corpus!')
            sentence_prob += math.log2(smoothed_trigram_prob)
        return sentence_prob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        corpus = list(corpus)
        log_prob_sums = 0
        corpus_num_words = sum([len(get_ngrams(sentence, 1)) for sentence in corpus])
        for sentence in corpus:
            log_prob_sums += self.sentence_logprob(sentence)

        return 2 ** ((-1)*(log_prob_sums / corpus_num_words))

    def get_training_corpus_perplexity(self):
        return self.perplexity(corpus_reader(self.training_corpusfile, self.lexicon))


if __name__ == "__main__":
    data_dir = './data'
    default_brown_train = f'{data_dir}/brown_train.txt'
    default_brown_test = f'{data_dir}/brown_test.txt'
    should_run_tests = False

    if should_run_tests:
        print('Running tests to ensure code is functioning...')
        run_tests(data_dir=data_dir, corpusfile_read_limit=200)
        print('Tests ran successfully!')
        print()

    print('Building model...')
    if len(sys.argv) < 2:
        model = TrigramModel(default_brown_train)
    else:
        model = TrigramModel(sys.argv[1])
    print('Model built!')
    print()

    print(f'Using model to generate sentence...')
    s = model.generate_sentence()
    print(f'Generated Sentence: {s}')
    print()

    print(f'Training Set Perplexity: {model.get_training_corpus_perplexity()}')

    if len(sys.argv) < 2:
        dev_corpus = corpus_reader(default_brown_test, model.lexicon)
    else:
        dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    print(f'Test Set Perplexity: {model.perplexity(dev_corpus)}')
    print()

    # Essay scoring experiment:
    print(f'Running essay scoring experiment...')
    acc = essay_scoring_experiment('./data/ets_toefl_data/train_high.txt', './data/ets_toefl_data/train_low.txt', './data/ets_toefl_data/test_high', './data/ets_toefl_data/test_low')
    print(f'Experiment Test Accuracy: {round(acc * 100, 5)}%')