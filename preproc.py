from collections import Counter

import pandas as pd
import numpy as np

# deliverable 1.1
def bag_of_words(text):
    """
    Count the number of word occurrences for a given document
    
    :param text: a document, as a single string
    :returns: a Counter representing a single document's word counts
    :rtype: Counter
    """
    return Counter(text.split())
   
# deliverable 1.2
def aggregate_counts(bags_of_words):
    """
    Aggregate bag-of-words word counts across an Iterable of documents into a single bag-of-words.
    
    :param bags_of_words: an iterable of bags of words, produced from the bag_of_words() function above
    :returns: an aggregated bag of words for the whole corpus
    :rtype: Counter
    """
    #print(bag_of_words[0])
    agg_counts = Counter()  
    for i in range(len(bags_of_words)):
        agg_counts += bags_of_words[i]
    return agg_counts
    
    
# deliverable 1.3
def compute_oov(bow1, bow2):
    """
    Return a set of words that appears in bow1, but not bow2

    :param bow1: a bag of words
    :param bow2: a bag of words
    :returns: the set of words in bow1, but not in bow2
    :rtype: set
    """
    return set(bow1) - set(bow2)
    
    
# deliverable 1.4
def prune_vocabulary(training_counts, target_data, min_counts):
    """
    Prune target_data to only include words that occur at least min_counts times in training_counts
    
    :param training_counts: aggregated Counter for the training data
    :param target_data: list of Counters containing dev bow's
    :returns: new list of Counters, with pruned vocabulary
    :returns: list of words in pruned vocabulary
    :rtype list of Counters, set
    """
    test = True
    k = 0
    pruned_counts = []
    pruned_words = set()
    for i in range(len(target_data)):
        new_counter = Counter()
        for w in target_data[i].items():
            if training_counts[w[0]] >= min_counts:
                new_counter[w[0]] = w[1]
                pruned_words.add(w[0])
        pruned_counts.append(new_counter)
  
    #print("PRUNED_COUNTS[95]: ")
    #print(sorted(pruned_counts[95].items()))
    return pruned_counts, pruned_words

# Helper functions

def read_data(fname, label='Era', preprocessor=bag_of_words): 
    df = pd.read_csv(fname)
    return (df[label].values, [preprocessor(string) for string in df['Lyrics'].values])
    
def oov_rate(bow1, bow2):
    return len(compute_oov(bow1, bow2)) / len(bow1.keys())