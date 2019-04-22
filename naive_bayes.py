from hw2_utils.constants import OFFSET
from hw2_utils import clf_base, evaluation

import numpy as np
from collections import defaultdict, Counter
from itertools import chain
import math

# deliverable 3.1
def get_corpus_counts(x,y,label):
    """
    Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """
    labcounts = defaultdict(lambda:0)
    for i in range(len(x)):
        if y[i] == label:
            for k, v in x[i].items():
                labcounts[k] += v
    return labcounts


# deliverable 3.2
def estimate_pxy(x,y,label,smoothing,vocab):
    """
    Compute smoothed log-probability P(word | label) for a given label. (eq. 2.30 in Eisenstein, 4.14 in J&M)

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param smoothing: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word

    """
    
    pxy = defaultdict(lambda:0.0)
    counts = get_corpus_counts(x, y, label) #returns dict of word:count appearances in total x data given label
    
   
    
    
    #i = 1
   
    #if OFFSET in vocab:
        #print("OFFSET in vocab in estimate_pxy")
     #   vocab.remove(OFFSET)
   
    #if OFFSET in counts.keys():
        #print("OFFSET IN COUNTS IN PXY, counts: ")
        #print(counts[OFFSET])
      #  counts[OFFSET] = 0
  
   
            
    for w in vocab:
        if w != OFFSET:
            pxy[w] = np.log((counts[w] + smoothing) / (sum(counts.values()) + (len(vocab) * smoothing)))
        else:
            pxy[w] = np.log((1 / sum(counts.values()))  * (len(y[y==label]) / len(y)))
            print("Here in estimate_ppxy")
        #print(sum(np.exp(list(pxy.values()))))
    
    #print("sentences w label: %s, %d " % (label, len(y[y==label])))
    #print("Out of %d sentences" % (len(y)))
    #print(len(vocab))
    #pxy[OFFSET] = np.log((1 / sum(counts.values()))  * (len(y[y==label]) / len(y)))
    return pxy

# deliverable 3.3
def estimate_nb(x,y,smoothing):
    """
    Estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights, as a default dict where the keys are (label, word) tuples and values are smoothed log-probs of P(word|label)
    :rtype: defaultdict 
    """ 
  
    #for every unique word in x (vocab), calc P(word|label)
    vocab = set(chain.from_iterable(x))
    
    
   # if OFFSET in vocab:
    #    print("OFFSET in vocab in estimate_nb  ----  1")
        
  #  print(vocab)
      
    for x_i in x:
        if OFFSET not in x_i.keys():
            x_i[OFFSET] = 1
    
    
    labels = list(set(y))
    word_label_probs = defaultdict(float)
    weights = defaultdict(float)
    
    label_probs = []
    for i in range(len(labels)):
        label_probs.append(estimate_pxy(x, y, labels[i], smoothing, vocab))
        if OFFSET in vocab:
            print("OFFSET in vocab in estimate_nb  ----  2")
        
        for w in vocab:
            word_label_probs[(labels[i], w)] = label_probs[i][w]
            
    return word_label_probs
    

# deliverable 3.4
def find_best_smoother(x_tr,y_tr,x_dv,y_dv,smoothers):
    """
    Find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param smoothers: list of smoothing values
    :returns: best smoothing value, scores
    :rtype: float, dict mapping smoothing value to score
    """
    scores = defaultdict(float)
    labels = list(set(y_dv))
    for s in smoothers:
        theta_nb = estimate_nb(x_tr, y_tr, s)
        y_ht = clf_base.predict_all(x_dv, theta_nb, labels)
        scores[s] = evaluation.acc(y_ht, y_dv)
    return max(scores, key=scores.get), scores
    
    