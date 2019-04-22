from nltk.util import ngrams


DIST_PEN = -2.0
MAX_DISTANCE = 4


class Derivation(object):
    def __init__(self, LM, PM, sentence):
        self.LM = LM
        self.PM = PM
        self.phrases = []
        self.source_words = []
        self.words = ['<bos>','<bos>']
        
        self.source_states = {}
        for word in sentence:
            self.source_states[word] = 0
            
        self.score = 0
        self.last_end = 0
        self.LM_score = 0
        self.PM_score = 0
        self.distance = 0
        self.VALID = True
        
        
        
        
    def add_score(self, phrase):
        
        lang_m_score = 0
        phrase_m_score = 0
        distance = 0
        words = []
        
        #get phrase model score
        self.PM_score += phrase.score
        self.distance += DIST_PEN * abs(self.last_end + 1 - phrase.start)

        if distance > MAX_DISTANCE:
            self.VALID = False
            print("Over max distance: %d , Max distance: %d " % (distance, MAX_DISTANCE))
            return 0

        for word in phrase.t_words:
            words.append(word)
            

        #get language model score
        for w1, w2, w3 in ngrams(self.words[-2:] + words, 3, pad_left=False):  #already have a pad
            self.LM_score += self.LM[(w1, w2)][w3]
            
        self.words += words
      
            
        return self.LM_score + self.PM_score + self.distance
    
    
    def add_phrase(self, phrase):
        self.phrases.append(phrase)
        p = phrase.score
       
        for word in phrase.s_words:
            if self.source_states[word] > 0:
                self.VALID = False
                print("Word: %s , already translated." % (word))
                return self.VALID
            #   
            else:
                self.source_states[word] = 1
                self.source_words.append(word)
                self.VALID = True
        
        self.score += self.add_score(phrase)
        self.last_end = phrase.end
        return self
    
  
        
    
    
    def decoded(self):
        return True if sum(self.source_states.values()) == len(self.source_states) else False
         
         
            
            
   
        
    def print_derivation(self):
        print("\nDERIVATION DATA")
        print('*' * 15)
        print("Phrases:")
        for p in self.phrases:
            p.print_phrase()
        print("\nWords")
        print(self.words)
        print("Source States")
        print(self.source_states)
        print("SCORE")
        print(self.score)
                
        
        