from IPython.display import HTML, display
from collections import defaultdict 
from tree import Tree
import numpy as np
import tabulate
import string
import re


#this fle contains functions for taking a file of syntax trees with root labels '(TOP'(otherwise some functions will need
#to be changed), and processes from them and returns a PCFG grammar dictionary, a reverse terminal dictionary for efficient
#reverse lookup, plain sentences recstructed from terminals (with punctuation spaced), and the original trees, which may be
#used or another PCFG with a sentence string to construct a parse tree using the CKY algorithm, if the sentence may be
#parsed using the given PCFG


#To use this CKY implementation requires at least one file containing a set of trees from a given grammar
#proc_trees() takes this file and collapses_unary() terminals, then transforms the trees into Chomski Normal Form,
#preserves/derives several pieces of information from the trees, including: 
#the probabilities of each rule -> terminal pair in a nested dictionary, 
#the counts of terminal : rule pairs.
#the set of terminals representing the original sentences,
#the trees,

#proc_trees() also accepts another boolean argument if the user wants single POS terminals collapsed, PT, which defaults to
#False, i.e.,

#with PT=True:
"""
(TOP
    (NP
        (DT the)
        (NN teacher)
    )
    (TOP|
        (VP
            (MD will)
            (VP
                (VB lecture)
                (VP|
                    (NP+NN today)
                    (PP
                        (IN in)
                        (NP
                            (DT the)
                            (NP|
                                (NN lecture)
                                (NN hall)
                            )
                        )
                    )
                )
            )
        )
        (. .)
    )
)

"""

#with PT=False
"""
(TOP
    (NP
        (DT the)
        (NN teacher)
    )
    (TOP|
        (VP
            (MD will)
            (VP
                (VB lecture)
                (VP|
                    (NP
                        (NN today)
                    )
                    (PP
                        (IN in)
                        (NP
                            (DT the)
                            (NP|
                                (NN lecture)
                                (NN hall)
                            )
                        )
                    )
                )
            )
        )
        (. .)
    )
)

"""
#the latter tree would be unparseable using the CKY implementation below


#CKY() takes a PCFG nested dictionary PCFG[gen][rule] = P(rule | gen) , and a plain sentence string separated by spaces (including punctuation, or 'word.' and 'word' will be different terminals), parses the most probable tree using Tree from tree.py, if it can be parsed, else outputs if terminal not in set or no 'TOP' root symbol in valid sentence generators in last row of table_syms.

#The last function is for displaying an html table if the user desires (be careful with output jams with large trees!)

    
    
#see description above..., reconstructs sentence by location terminals with 2 regular expressions: one for skipping blank lines, the other for the string to be captured.
def proc_trees(filename, PT=False):
    
                  #terminal             #generator         #occurences of pair
    term_counts = defaultdict(lambda: defaultdict (lambda: 0.0))
    gen_counts = defaultdict(lambda: defaultdict (lambda: 0.0))
    parens = ['(', ')']
    new = True
    tree_string = "(TOP"
    original_sen = ""
    og_sens = []
    trees = []
 
    
    for level in open(filename, 'r'):   
        level = level.strip()
        
        #try to get original sentence (all lower or punct terminals)
        if level not in parens:
            if re.match(r'[^\s]', level): #make sure it's not a blank space
            
                #get word for sentence
                word_level = level.strip('(').strip(')').split()
                word = word_level[max(0, len(word_level) - 1)]
                if re.match(r'([A-Z"]*[a-z"]+)|([a-z0-9"]+)|([\'`,";:?*%$!\.]+)|(<[A-Z]*>)', word):
                    original_sen += ' ' + word

        if level != '(TOP':
            tree_string += level if level != ')' else ' ' + level
            new = False

        elif level == '(TOP' and new == False:
            og_sens.append(original_sen)
            original_sen = ""
            tree_string = '' + tree_string + ''
            t = Tree.from_string(tree_string)
            t.collapse_unary(POS_TERM=PT).chomsky_normal_form()
            trees.append(t)
            term_counts, gen_counts = cfg_counts(t, term_counts, gen_counts)
            tree_string = level
                
            
            
    return get_probs(gen_counts), term_counts, og_sens, trees




#given a Tree object, recursively returns 2 reverse dictionaries of term:gen, gen:term counts
def cfg_counts(cfg_tree, term_counts=False, gen_counts=False):
    
    term_counts = term_counts if term_counts else defaultdict(lambda: defaultdict (lambda: 0.0))
    gen_counts = gen_counts if gen_counts else defaultdict(lambda: defaultdict (lambda: 0.0))
        
    #store counts of label sequences seen as tuples in a dictionary
    if hasattr(cfg_tree, 'label'):
        dlist = [d.label if hasattr(d, 'label') else d for d in cfg_tree.daughters]
        term_counts[' '.join(dlist)][cfg_tree.label] += 1   #made to tuple bc list unhashable 
        gen_counts[cfg_tree.label][' '.join(dlist)] += 1 
        
        
        #recurse into each daughter
        for d in cfg_tree.daughters:
            cfg_counts(d, term_counts, gen_counts)
        
                
    return term_counts, gen_counts

#takes a dictionary of counts, returns nested probability of gen -> term of P(gen | term)
def get_probs(gen_counts):
    
    #compute probs of #terminal   :   #mother pairs             
    gen_probs = defaultdict(lambda: defaultdict(lambda: 0.0))
    for gen, d in gen_counts.items():
        for term, v in d.items():
            gen_probs[gen][term] = v / sum(d.values())
   
    return gen_probs





#gets all sym1 + sym2 combos for sym1 in init_syms and sym2 in final_syms
def get_combos(init_syms, final_syms):
    combos = [] 
    for sym1 in init_syms:
        for sym2 in final_syms:
            combos.append([sym1, sym2])
    return combos


#returns a list of valid generators for given terminal if found in dictionary
def get_generators(term_counts, terminal):
    return [gen for gen in term_counts[terminal].keys() if terminal in term_counts.keys()]
    
    
#utility object for storing and carrying concatenated labels 
#and highest probability of a given root at a place in the tree
class Table_Sym:
    def __init__ (self, path_string, path_prob):
        self.path_string = path_string
        self.path_prob = path_prob
        
    def print_sym(self):
        print(f"Table_Sym path_string: {self.path_string}")
        print(f"Table_Sym prob: {self.path_prob}")
        
        
#reconstructs parse tree from labels of table if non-terminal starting root 'TOP' in top row of table,
#outputs error message, returns None if starting root not found
#change 'TOP' to other starting root symbol if needed (e.g. 'S')
def get_parse_tree(cky_table):
   
    if 'TOP' in cky_table[-1][0].keys():
        cky_table[-1][0]['TOP'].print_sym()
        tree_string = cky_table[-1][0]['TOP'].path_string
        return Tree.from_string(tree_string)
    else:
        print("Unparseable, TOP not in sentence generators.")
        return None
   
 
#takes plain string of sentence separated by spaces, including punctuation, splits into list of words,
#intializes first row by creating a dictionary in each cell whose keys are every possible generator for the given sequence
#combo of symbols being considered by the cell. For each generator, if the given sequence (subtree) can be created by more
#than one combination, it stores only the higher probability subtree in an object as the value of the cell dictionary for 
#the given generator key, with the string of the subtree concatenated in a way to be given in the end to
#tree.from_string(tree_string), in order to recreate the parse tree if possible.
def CKY(term_counts, gen_probs, sentence):
    
    sentence = [word for word in sentence.split()]
    table_syms = [[] * len(sentence)] #a list of lists (matrix)  
                            
                                                 
    #initialize first row
    first_row = []
    for word in sentence:
        cell_sym_paths = defaultdict(lambda:Table_Sym)
        for gen in get_generators(term_counts, word):
            cell_sym_paths[gen] = Table_Sym('(' + gen + ' ' + word + ')', np.log(gen_probs[gen][word])) 
        
        if len(cell_sym_paths) == 0:
            print(f"Sentence unparseable, contains unseen terminal(s): {word}")
            return None, None, None
        
        first_row.append(cell_sym_paths)               
    table_syms[0] = first_row
    
 
    for i in range(1, len(sentence)):
        for j in range(len(sentence) - i):
            cell_sym_paths = defaultdict(lambda:Table_Sym)
            
            #get all grammatical combinations of left + right subsets of sentence syms   
            for k in range(i):    
                combos = get_combos(table_syms[k][j].keys(), table_syms[i - (k + 1)][j + (k + 1)].keys())
                for c in combos:
                    
                    #i.e. g -> c is a valid rule in grammar   
                    valid_gens = get_generators(term_counts, ' '.join(c))
                    for g in valid_gens: 
                        subpath_string = table_syms[k][j][c[0]].path_string +table_syms[i-(k+1)][j+(k+1)][c[1]].path_string
                        subpath_prob = table_syms[k][j][c[0]].path_prob + table_syms[i-(k+1)][j+(k+1)][c[1]].path_prob
                        cell_sym =   Table_Sym (
                                                path_string = '(' + g + ' ' + subpath_string + ')', 
                                                path_prob = np.log(gen_probs[g][' '.join(c)]) + subpath_prob 
                                     )
                        
                        #only keep the max combo subtree for a given sym root
                        if g in cell_sym_paths.keys():    
                            if cell_sym_paths[g].path_prob < np.log(gen_probs[g][' '.join(c)]) + subpath_prob:
                                 cell_sym_paths[g] = cell_sym
                        else:
                            cell_sym_paths[g] = cell_sym
                   
            if len(cell_sym_paths.keys()) == 0:
                cell_sym_paths['_'] = Table_Sym('_', 0)
        
            #add to table
            if len(table_syms) != i:
                table_syms[i].append(cell_sym_paths) 
            else:
                table_syms.append([cell_sym_paths])
   
         
    return table_syms, True if 'TOP' in table_syms[-1][0].keys() else False, get_parse_tree(table_syms)



def html_table(table):
    table_labels = []
    for row in table:
        c = []
        for cell in row:
            c.append([k for k in cell.keys()])
        
        table_labels.append(c)
    
    display(HTML(tabulate.tabulate(table_labels, tablefmt='html')))
   
            
         