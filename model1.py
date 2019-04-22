#! export PYTHONIOENCODING=UTF-8
#!/usr/bin/env python
# model1.py: Model 1 translation table estimation
# Steven Bedrick <bedricks@ohsu.edu> and Kyle Gorman <gormanky@ohsu.edu>

from collections import defaultdict
import numpy as np

def bitext(source, target):
    """
    Run through the bitext files, yielding one sentence at a time
    """
    for (s, t) in zip(source, target):
        yield (s.strip().split(), [None] + t.strip().split())
        # by convention, only target-side tokens may be aligned to a null
        # symbol (here represented with None) on the source side


def get_sentences(s_filename, t_filename):
    s = open(s_filename, 'r')
    t = open(t_filename, 'r')

    s_sens, t_sens = [], []
    
    I = 0
    for source_sen, target_sen in bitext(s, t):
        if I < 2000:
            s_sens.append(source_sen)
            t_sens.append(target_sen)
        I += 1
       
        
    
    return s_sens, t_sens

class Model1(object):
    """
    IBM Model 1 translation table
    """

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

    def __init__(self, source_file, target_file):
        self.source, self.target = get_sentences(source_file, target_file)
        self.trans_probs, self.t_vocab = self.get_initial_probs() 
 
    
    def get_initial_probs(self):
        trans_probs = defaultdict(lambda: defaultdict(lambda:0.0)) #alignment table dictionary
        co_counts = defaultdict(lambda: defaultdict(lambda:0.0))   #for sentence co-occurence counts
        t_vocab = defaultdict(lambda:0)                            #e_vocab will be the keys of the translation dictionary

    
        #first get co-occurrence counts, vocabs
        for s, t in zip(self.source, self.target):
            for word_s in s:
                for word_t in t:
                    t_vocab[word_t] += 1
                    co_counts[word_s][word_t] += 1  
            
            

        #create translation dictionary, intitialize to uniform distribution
        #only initialize probs for words that appeared in paired sentences
        
        V = len(t_vocab)
        for word_s in co_counts.keys():
            for word_t in co_counts[word_s].keys():
                trans_probs[word_s][word_t] = 1.0 / V  
       
        return trans_probs, t_vocab
    
    
    
    
    
    
    def em_ibm1(self):
        
        #borrowed heavily from pseudocode in Dr. P. Koehn's lecture: http://mt-class.org/jhu/slides/lecture-ibm-model1.pdf
        new_counts = defaultdict(lambda:defaultdict(lambda:0.0))
        total_sen = defaultdict(lambda:0.0)
        total_s = defaultdict(lambda:0.0)
        
    
        #get word alignment p(t|s)
        for s, t in zip(self.source, self.target):
            
            for word_t in t:
                total_sen[word_t] = 0
                for word_s in s:
                    total_sen[word_t] += self.trans_probs[word_s][word_t]
                 
            for word_t in t:
                for word_s in s:
                    new_counts[word_s][word_t] += self.trans_probs[word_s][word_t] / total_sen[word_t]
                    total_s[word_s] += self.trans_probs[word_s][word_t] / total_sen[word_t]  
              
      
        for word_s in total_s.keys():
            for word_t in self.trans_probs[word_s].keys():
                self.trans_probs[word_s][word_t] = new_counts[word_s][word_t] / total_s[word_s]
        
        print("AFTER EM")
        return self.trans_probs
    
    
    #Perform n iterations of EM training
    def train(self, n):
       
        for i in range(n):
            self.em_ibm1()
        
        return self.make_dict(), self.trans_probs

    
    
    def make_dict(self):
        
        ts_dictionary = defaultdict(lambda:defaultdict(lambda:0.0))
        for word_s in self.trans_probs.keys():
            most_prob_t = max(self.trans_probs[word_s], key=lambda key: self.trans_probs[word_s][key])
            ts_dictionary[word_s][most_prob_t] = self.trans_probs[word_s][most_prob_t]
           
        return ts_dictionary 


if __name__ == '__main__':
    import doctest
    doctest.testmod()



