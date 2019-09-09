import model1
from random import random
from collections import defaultdict

ITER_MAX = 5




class Model2(object):
    
    def __init__(self, source_file, target_file, trans_probs=False, n=False):
        if trans_probs:
            self.trans_probs = trans_probs
        elif n:
            _, self.trans_probs = model1.Model1(source_file, target_file).train(n)
        else:
            _, self.trans_probs = model1.Model1(source_file, target_file).train(ITER_MAX)
            
        self.source, self.target = model1.get_sentences(source_file, target_file)
        self.dist_probs = self.get_init_dist_probs()
        
    def get_init_dist_probs(self):
        
        #prob of word in i pos of source aligned with word in j pos of target, given target sen len l, source sen len m
        dist_probs = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:0.0))))
        
        s_lengths = [len(s) for s in self.source]
        t_lengths = [len(t) for t in self.target]
        
        
        for m, l in zip(s_lengths, t_lengths):
            for j in range(m):
                for i in range(l):
                    dist_probs[j][i][m][l] = random()
                    while dist_probs[j][i][m][l] == 0:     #rare but possible, could just define range...
                        dist_probs[j][i][m][l] = random()
                            
        return dist_probs
    
    
    
    
    #performs a single iteration of the IBM Model2 translation-alignment algorithm
    def em_ibm2(self):
        
        #counts of co-occurrences of e given f in respective target, source sentences 
        ts_counts = defaultdict(lambda:defaultdict(lambda:0.0))
        
        #prob of word in j pos of source aligned with word in i pos of target, given target sen len l, source sen len m
        dist_counts = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:0.0))))
        dist_pos_counts = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:0.0)))
      
        total_sen = defaultdict(lambda:0.0)
        total_s = defaultdict(lambda:0.0)
        
        
        #E-step, compute expected counts of alignments parameterized by word alignment probs and distortion probs
        for t, s in zip(self.target, self.source):
                    
            m = len(s)
            l = len(t)
            for i, word_t in enumerate(t):
                total_sen[word_t] = 0
                for j, word_s in enumerate(s):
                    total_sen[word_t] += self.trans_probs[word_s][word_t] * self.dist_probs[j][i][m][l]
                    
        
            #compute fractional counts       
            for i, word_t in enumerate(t):
                for j, word_s in enumerate(s):
                    
                  
                    if total_sen[word_t] > 0:
                        d = (self.trans_probs[word_s][word_t] * self.dist_probs[j][i][m][l]) / total_sen[word_t]
                    else:
                        d = 0   #lazy smoothing
                        
                        
                    ts_counts[word_s][word_t] += d
                    total_s[word_s] += d
                    dist_counts[j][i][m][l] += d
                    dist_pos_counts[j][m][l] += d
           
        #M-step  
        
        #for word alignment
        for word_s in total_s.keys():
            for word_t in self.trans_probs[word_s].keys():
                self.trans_probs[word_s][word_t] = ts_counts[word_s][word_t] / total_s[word_s]
                
                
                
        #for distortion
        s_lengths = [len(f) for f in self.source]
        t_lengths = [len(e) for e in self.target]
        
        for m, l in zip(s_lengths, t_lengths):
            for j in range(m):
                for i in range(l):
                    self.dist_probs[j][i][m][l] = dist_counts[j][i][m][l] / dist_pos_counts[j][m][l]


        return self.trans_probs, self.dist_probs
       
        
     
    #returns dictionay of f : argmax(p(e | f))
    def make_dict(self):
        ts_dictionary = defaultdict(lambda:defaultdict(lambda:0.0))
        for word_s in self.trans_probs.keys():
            most_prob_t = max(self.trans_probs[word_s], key=lambda key: self.trans_probs[word_s][key])
            ts_dictionary[word_s][most_prob_t] = self.trans_probs[word_s][most_prob_t]
           
        return ts_dictionary 



        
        
        
    #Perform n iterations of EM training
    def train(self, n):
       
        for i in range(n):
            self.em_ibm2()
            print("Next epoch")
        
        
        return self.make_dict(), self.trans_probs, self.dist_probs
        
     
    #for every word in every source sentence, get most probable alignment word in target sentence,
    #return dictionary of this sentence alignment
    def get_sen_alignment(self, s_sen, t_sen):
        
        alignments = defaultdict(lambda: str)
        
        m = len(s_sen)
        l = len(t_sen) 
        for j, word_s in enumerate(s_sen):
            
            a_word = ""
            a_prob = 0.0
            for i, word_t in enumerate(t_sen):
                ta_prob = self.trans_probs[word_s][word_t] * self.dist_probs[j][i][m][l]
                if ta_prob > a_prob:
                    a_word = word_t
                    a_prob = ta_prob
            #print(word_s, a_word)
            alignments[word_s] = a_word
            
            
        return alignments
          
        
        
    #returns a list of dictionaries giving most probable f | e (and position given len, 
    #calculated in em_ibm2 for every f, e in for every sentence pair        
    def get_all_alignments(self):
          
        all_aligns = []
        for s, t in zip(self.source, self.target):
            all_aligns.append(self.get_sen_alignment(s, t))
        
        return all_aligns


    
    
   
        
        
        
        
        