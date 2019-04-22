from collections import defaultdict
from nltk.util import ngrams
from math import log
import derivation


        
class Phrase(object):
    def __init__(self, start, s_words, t_words, score):
        self.start = start
        self.s_words = s_words
        self.t_words = t_words
        self.score = score
        self.end = start + len(t_words)
        
    def print_phrase(self):
        print("PHRASE DATA:")
        print("S_Words:")
        print(self.s_words)
        print("T_Words:")
        print(self.t_words)
        print("Score:")
        print(self.score)
        print("Start")
        print(self.start)
        



class PhraseModel(object):

    def __init__(self, st_aligns, ts_aligns, s_filename, t_filename):
        self.s_sens = [s.strip('\n').split() for k, s in enumerate(open(s_filename, 'r').readlines()) if k < 250000]
        self.t_sens = [t.strip('\n').split() for k, t in enumerate(open(t_filename, 'r').readlines()) if k < 250000]
        self.st_aligns = st_aligns
        self.ts_aligns = ts_aligns
        self.sen_overlaps_st, self.sen_overlaps_ts = self.get_align_overlaps()
        self.phrase_lex = {}
        self.MAX_PHRASE_LENGTH = 4
        self.MIN_PHRASE_COUNT = 1
        self.trigram_model_t = {}
        self.BEAM = 5
       
    

    
    #outputs single dictionary of f : e alignments based on overlap of reciprocal models
    #using both source-to-target and target-to-source alignments from ibm model2 output, 
    def get_align_overlaps(self):

        sen_over_st = []
        sen_over_ts = []
        for st_sen_align, ts_sen_align in zip(self.st_aligns, self.ts_aligns):
            align_matrix_st, align_matrix_ts = defaultdict(lambda:[]), defaultdict(lambda:[])          
            for word_s in st_sen_align.keys():
                word_t = st_sen_align[word_s]
              
                #only have to check one direction since both necessary
                if ts_sen_align[word_t] == word_s:      
                    align_matrix_st[word_s].append(word_t)
                    align_matrix_ts[word_t].append(word_s)
                    

            sen_over_st.append(align_matrix_st)
            sen_over_ts.append(align_matrix_ts)

        return sen_over_st, sen_over_ts

    
    #borrowed mostly from Philip Koehn's: http://mt-class.org/jhu/slides/lecture-advanced-alignment-models.pdf
    def grow_alignments(self):
       
        neighboring = ((-1,0),(0,-1),(1,0),(0,1),(-1,-1),(-1,1),(1,-1),(1,1))
        
        for k, (sen_s, sen_t) in enumerate(zip(self.s_sens, self.t_sens)):
            
            if k % 10000 == 0:
                print(k)
                
                
            new_points_added = True
            st_overlap = self.sen_overlaps_st[k]
            ts_overlap = self.sen_overlaps_ts[k]
            st_aligns = self.st_aligns[k]
            ts_aligns = self.ts_aligns[k]
            
            
            while(new_points_added):
                new_points_added = False   #reset flag
                
                for i, word_s in enumerate(sen_s):
                    for j, word_t in enumerate(sen_t):
                        
                        #if k == 125:
                         #   print("\nword_s, word_t:")
                         #   print(word_s, word_t)
                            #print("ST overlap[word_s]: %s " % (st_overlap[word_s]))
                            
                        if word_s in st_overlap.keys() and word_t in st_overlap[word_s]:
                            
                            
                            #if k == 125:
                             #   print("\nChecking Neighbors:")
                         
                        
                            #CHECKING NEIGHBORS"
                            for (x, y) in neighboring:       #check for boundaries (otherwise it goes to the other side)
                                if i + x >= 0 and i + x < len(sen_s) and j + y >= 0 and j + y < len(sen_t):
                                    w_s = sen_s[i + x]
                                    w_t = sen_t[j + y]
                                    
                                    #if k == 125:
                                       # print("In Grow, neighbor w_s, w_t:")
                                       # print(w_s, w_t)
                                        
                                    st_vals = [v for l in st_overlap.values() for v in l]   
                                    ts_vals = [v for l in ts_overlap.values() for v in l]
                                   
                                    if w_t not in st_vals or w_s not in ts_vals:
                                        
                                       
                                        if w_s in st_aligns.keys() and st_aligns[w_s] == w_t:
                                           # if k == 125:
                                               # print("*****Growing!!!!*****")
                                               # print(w_s, w_t)
                                            st_overlap[w_s].append(w_t)
                                            ts_overlap[w_t].append(w_s)
                                            new_points_added = True

                                        elif w_t in ts_aligns.keys() and ts_aligns[w_t] == w_s:
                                           # if k == 125:
                                            #    print("*****Growing!!!!*****")
                                             #   print(w_s, w_t)
                                            ts_overlap[w_t].append(w_s)
                                            st_overlap[w_s].append(w_t)
                                            new_points_added = True

                                    
            self.sen_overlaps_st[k] = st_overlap
            self.sen_overlaps_ts[k] = ts_overlap
                              
         
                            
                            
    #returns true if for every word_s in s_phrase there is 
    #no aligned word outside of the words in t_phrase
    def is_consistent(self, s_phrase, t_phrase, st_overlap, ts_overlap):
         
        for word_s in s_phrase:
            for word_t in st_overlap[word_s]:
                if word_t not in t_phrase:
                    return False
                
        for word_t in t_phrase:
            for word_s in ts_overlap[word_t]:
                if word_s not in s_phrase:
                    return False
    
        return True
            
        
                    
                
                
                
                
    #returns all consistent phrase pairs from corpus as dictionary with counts of each occurence
    def get_phrase_lex(self):
        
        phrase_lex = defaultdict(lambda:defaultdict(lambda:0.0))
        
        for k, (s_sen, t_sen) in enumerate(zip(self.s_sens, self.t_sens)):
              
            if k % 1000 == 0:
                print(k)  
            
            st_o = self.sen_overlaps_st[k]
            ts_o = self.sen_overlaps_ts[k]
            T_start = 0                     #for keeping efficient target sentence idx, building consistent phrase pairs
            
            #for every source as phrase start...
            for i, word_s in enumerate(s_sen):
                
                s_phrase = [word_s]
                t_phrase = []
                phrase_length = 0
                
                #and every possible target combination, accumulatively...
                for word_t in t_sen[T_start:min(T_start + self.MAX_PHRASE_LENGTH, len(t_sen))]:
                    t_phrase.append(word_t)
            
                    if self.is_consistent(s_phrase, t_phrase, st_o, ts_o):
                        phrase_lex[tuple(s_phrase)][tuple(t_phrase)] += 1
                        phrase_length = len(t_phrase)
                        break                     #no 2 phrases can be consistent in the same column
                    
                for next_word in s_sen[i + 1:min(i + 1 + self.MAX_PHRASE_LENGTH, len(s_sen))]:
                    s_phrase.append(next_word)
                    
                    t_phrase = []
                    for word_t in t_sen[T_start:min(T_start + self.MAX_PHRASE_LENGTH, len(t_sen))]:
                        t_phrase.append(word_t)
                        if self.is_consistent(s_phrase, t_phrase, st_o, ts_o):
                            phrase_lex[tuple(s_phrase)][tuple(t_phrase)] += 1
                            break                   #no 2 phrases can be consistent in the same row either
                 
                T_start += phrase_length
                
        self.phrase_lex = phrase_lex                
        return phrase_lex
    
    
    
    def phrase_lex_to_probs(self):
        
        phrase_lex_probs = defaultdict(lambda:defaultdict(lambda:0.0))
        
        for word_s, word_t_dict in self.phrase_lex.items():
            count_f = sum(word_t_dict.values())
            for word_t in word_t_dict.keys():
                if word_t_dict[word_t] > self.MIN_PHRASE_COUNT:
                    phrase_lex_probs[word_s][word_t] = log(word_t_dict[word_t] / count_f)
                    
                    
        self.phrase_lex = phrase_lex_probs
        return
    
    #uses nltk.utils.ngram to build trigram language model of p(w3 | w1w2)
    def build_trigram_model(self):
        
        trigram_model = defaultdict(lambda:defaultdict(lambda:0.0))
        
        #get counts first
        for t_sen in self.t_sens:
            for w1, w2, w3 in ngrams(t_sen, 3, pad_left=True, left_pad_symbol='<bos>'):
                trigram_model[(w1, w2)][w3] += 1
            
        
        self.trigram_model_t = self.trigram_counts_to_probs(trigram_model)
        return self.trigram_model_t
    
    
    def trigram_counts_to_probs(self, trigram_model):
        
        trigram_probs = defaultdict(lambda:defaultdict(lambda:0.0))
        
        for word_s, word_t_dict in trigram_model.items():
            count_f = sum(word_t_dict.values())
            for word_t in word_t_dict.keys():
                trigram_probs[word_s][word_t] = log(word_t_dict[word_t] / count_f)
                    
                   
        return trigram_probs
    
        


    def beam(self, top_values):

        beam_values = []
        beam_range = min(len(top_values), BEAM_SIZE)
        top_value = top_values[0]

        if len(top_values) > 0:
            for v in top_values[:beam_range]:
                if abs(v - top_value) < MAX_PROB_DIFF:
                    beam_values.append(v)

        return beam_values


    def decode_sentence(self, source_sentence):

        source_sentence = source_sentence.split()

        derivation_stack = [derivation.Derivation(self.trigram_model_t, self.trigram_model_t, source_sentence)]
        possible_derivations = []

        #until the stack is empty 
        while(len(derivation_stack) > 0):

            new_derivation_stack = []
            for d in derivation_stack:

                new_hyps = []
                poss_phrase = tuple()
                for i, s_word in enumerate(source_sentence, 1):
                    if d.source_states[s_word] == 0:
                        poss_phrase += (s_word,)
                        top_values = sorted(P_MT.phrase_lex[poss_phrase].values())[::-1]

                        #BEAM narrow the possible matches
                        if len(top_values) > 0:
                            beam_values = self.beam(top_values)
                        else:
                            beam_values = []


                        t_pos = [t for t in self.phrase_lex[poss_phrase].keys() if self.phrase_lex[poss_phrase][t] in beam_values]    

                        for target_phrase in t_pos:
                            pm_s = self.phrase_lex[poss_phrase][target_phrase]


                            V = True
                            for word in poss_phrase:
                                if d.source_states[word] > 0:
                                    V = False


                            if V:

                                state_match = False
                                temp_d = derivation.Derivation(self.trigram_model_t, self.trigram_model_t, source_sentence)
                                for phrase in d.phrases:
                                    temp_d.add_phrase(phrase)

                                temp_d.add_phrase(Phrase(start=i, s_words=poss_phrase, t_words=target_phrase, score=pm_s))        

                                for i, hyp in enumerate(new_hyps):  

                                    if temp_d.source_states == hyp.source_states:
                                        state_match = True
                                        if temp_d.score > hyp.score:
                                            new_hyps[i] = temp_d

                                #if we didn't replace a lower prob derivation
                                if state_match == False:
                                    new_hyps.append(temp_d)


                                if temp_d.decoded():
                                    possible_derivations.append(temp_d)


                new_derivation_stack += new_hyps

            derivation_stack =  new_derivation_stack


        m_score = 999999
        m_words = []
        for d in possible_derivations:
            if d.score < m_score:
                m_score = d.score
                m_words = d.words



        return [(m_words[2:], m_score)]




















