import string
import nltk
import nltk.data
import numpy as np
import torch
import torch.nn as nn
import random
import itertools
from matplotlib import pyplot as plt
from .data import Observation
from hw4_utils import data, vocab
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score


PUNKT_DETECTOR = nltk.data.load('tokenizers/punkt/english.pickle')

def baseline_classifier(obs):
    """
    Baseline: Always say it's a sentence break
    """
    return True
    

def next_tok_capitalized_baseline(obs):
    """
    True if the right token is caplitalized
    """
    return obs.right_token[0] in string.ascii_uppercase

def punkt_baseline(obs):
    """
    Use the NLTK pre-trained Punkt classifier to try and decide if our candidate is a sentence break.
    
    This is not _exactly_ fair to Punkt, as it is not designed to work on fragments like this, but it's OK.
    """
    return len(PUNKT_DETECTOR.sentences_from_text(obs.orig_obs)) > 1






# First Sentence Classifier, Naive Bayes 

#naive bayes / heuristic sentence boundary classifier generally based on example in Bird, Klein & Loper (2009, p. 234)
#8 features based on intuition and a smidgeon of research :)
def sen_features(obs):
    return {'next-word-capitalized':obs.right_token[0].isupper(), 'next-char-quote':obs.right_token[0]!='"', 'prevword':obs.left_token, 'leftlength':len(obs.left_token)> 2, 'left_not_digit':obs.left_token[-1].isdigit() == False, 'digit_left_no_digit_right':obs.left_token[-1].isdigit() and (obs.right_token[0].isdigit() == False),
  'punct':obs.punctuation_mark, 'left_not_co':obs.left_token!='Co'}


def feature_sets(obs_list, labels):
    return [(sen_features(x), labels[i]) for i, x in enumerate(obs_list)]

def nb_sen_classifier(fileroot):
    
    #loads text from file in special way using data.load_candidates to return namedtuple of features for classification
    #assumes data is already tri-partitioned into train, dev, test
    x_tr = [o for o in data.load_candidates(data.file_as_string(fileroot + "/train.txt"))]
    y_tr =  [o.is_true_break for o in data.load_candidates(data.file_as_string(fileroot + "/train.txt"))]

    x_test = [o for o in data.load_candidates(data.file_as_string(fileroot+ "/test.txt"))]
    y_test =  [o.is_true_break for o in data.load_candidates(data.file_as_string(fileroot + "/test.txt"))]

    x_dev = [o for o in data.load_candidates(data.file_as_string(fileroot + "/dev.txt"))]
    y_dev = [o.is_true_break for o in data.load_candidates(data.file_as_string(fileroot + "/dev.txt"))]


    train_feat_sets = feature_sets(x_tr, y_tr)
    test_feat_sets = feature_sets(x_test, y_test)
    dev_feat_sets = feature_sets(x_dev, y_dev)

    boundary_detector = nltk.NaiveBayesClassifier.train(train_feat_sets)
    
    print(nltk.classify.accuracy(boundary_detector, test_feat_sets))
    return boundary_detector, x_test, y_test















# Second Sentence Classifier, RNN w LSTM

#utility function to make output comparison more clear than bools
def rename_labels(labels):
    labs = []
    for l in labels:
        if l:
            labs.append("True sentence.")
        else:
            labs.append("False sentence.")
    return labs



def rnn_sen_classifier(fileroot):
    
    #pre-process data into usable format for model
    left_sents_tr = [o.left_raw + o.punctuation_mark for o in data.load_candidates(data.file_as_string(fileroot + "/train.txt"))]
    labels_tr = [o.is_true_break for o in data.load_candidates(data.file_as_string(fileroot + "/train.txt"))]

    left_sents_dev = [o.left_raw + o.punctuation_mark for o in data.load_candidates(data.file_as_string(fileroot + "/dev.txt"))]
    labels_dev = [o.is_true_break for o in data.load_candidates(data.file_as_string(fileroot + "/dev.txt"))]


    left_sents_test = [o.left_raw + o.punctuation_mark for o in data.load_candidates(data.file_as_string("data/wsj_sbd/test.txt"))]
    labels_test = [o.is_true_break for o in data.load_candidates(data.file_as_string(fileroot + "/test.txt"))]


    Obs_Test = [o for o in  data.load_candidates(data.file_as_string("data/wsj_sbd/dev.txt"))]
    og_sents_test =  [o.orig_obs for o in data.load_candidates(data.file_as_string(fileroot + "/test.txt"))]

    
    labels_tr = rename_labels(labels_tr) 
    labels_test = rename_labels(labels_test) 
    labels_dev = rename_labels(labels_dev) 
    
    #prepare data for embedding layer
    wsj_c2i, wsj_i2c = vocab.build_vocab(left_sents_tr)
    wsj_l2i, wsj_i2l = vocab.build_label_vocab(labels_tr)

    train_data = (left_sents_tr, labels_tr)
    test_data = (left_sents_test, labels_test)
    dev_data = (left_sents_dev, labels_dev)

    #build an RNN with 50 embedding dims and 10 hidden dims, 1 lstm layer, 1 linear layer, 
    #and a LogSoftMax output layer to 2 classes, not modifiable because other configs were unstable (did not converge) for task.
    snn = SenNN(
    input_vocab_n=len(wsj_c2i),
    embedding_dims=50,
    hidden_dims=10,
    lstm_layers=1,
    output_class_n=2
    )
        
    
    #train the model
    train_model(
        model=snn,
        n_epochs=1,
        training_data=train_data,
        c2i=wsj_c2i, i2c=wsj_i2c,
        l2i=wsj_l2i, i2l=wsj_i2l
    );
        
    acc_snn, y_hat_snn = eval_acc(snn, test_data, wsj_c2i, wsj_i2c, wsj_l2i, wsj_i2l)     
    print(f"Accuracy: {acc_snn}")
    
    y = test_data[1]
    print(classification_report(y, y_hat_snn))
    return snn, y_hat_snn, y, Obs_Test, wsj_c2i, wsj_i2l
        



# pytorch RNN with lstm, linear, then logsoftmax output layers for classifying sentences, optimized for [left_text+punctuation] input
class SenNN(nn.Module):
    def __init__(self, input_vocab_n, embedding_dims, hidden_dims, lstm_layers, output_class_n):
        super(SenNN, self).__init__()
        
        # Saving this so that other parts of the class can re-use it
        self.lstm_dims = hidden_dims
        self.lstm_layers = lstm_layers
        
        self.input_lookup = nn.Embedding(num_embeddings=input_vocab_n, embedding_dim=embedding_dims)
        self.lstm = nn.LSTM(input_size=embedding_dims, hidden_size=hidden_dims, num_layers=lstm_layers, batch_first=True)
        
        # The output softmax classifier: first, the linear layer:
        self.output = nn.Linear(in_features=hidden_dims, out_features=output_class_n)
        
        # Then, the actual log-softmaxing:
        # Note that we are using LogSoftmax here, since we want to use negative log-likelihood as our loss function.
        self.softmax = nn.LogSoftmax(dim=2)
        
    # Expects a (1, n) tensor where n equals the length of the input sentence in characters
    # Will return a (output_class_n) tensor- one slot in the first dimension for each possible output class
    def forward(self, sentence_tensor):
        x = self.input_lookup(sentence_tensor)
        y = self.lstm(x)[0][-1]
        y = self.output(y).unsqueeze(0)   
        return self.softmax(y).squeeze()[-1]
        

   #random noise for first hidden layer values
    def init_hidden(self):
        h0 = torch.randn(self.lstm_layers, 1, self.lstm_dims)
        c0 = torch.randn(self.lstm_layers, 1, self.lstm_dims)
        return (h0, c0)

def predict_one(model, s, c2i, i2l):
    """
    Runs a sentence, s, through the model, and returns the predicted label.
    
    Make sure to use "torch.no_grad()"!
    See https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#gradients for discussion
    
    :param model: The LangID model to use for prediction
    :param s: The sentence to pss through, as a string
    :param c2i: The dictionary to use to map from character to index
    :param i2l: The dictionary for mapping from output index to label
    :returns: The predicted label for s
    :rtype: str
    """
    
    with torch.no_grad():
        p = model(vocab.sentence_to_tensor(s, c2i))
        
    return i2l[p.exp().argmax().item()]
    

def eval_acc(model, test_data, c2i, i2c, l2i, i2l):
    """
    Compute classification accuracy for the test_data against the model.
    
    :param model: The trained model to use
    :param test_data: A Pandas dataframe, containing a set of sentences to evaluate. Will include columns named "lang" and "sentence"
    :returns: The classification accuracy (n_correct / n_total), as well as the predictions
    :rtype: tuple(float, list(str))
    """
    
    y_hat = []
    y_preds = []
    for i, s in enumerate(test_data[0]): 
        y_preds.append(predict_one(model, s, c2i, i2l))
        if y_preds[i] == test_data[1][i]:
            y_hat.append(1)
        else: 
            y_hat.append(0)
            
    
    return (sum(y_hat) / len(y_hat) , y_preds)
   

def train_model(model, n_epochs, training_data, c2i, i2c, l2i, i2l):
    """
    Train using the Adam optimizer.
    
    :returns: The trained model, as well as a list of average loss values from during training (for visualizing) loss stability, etc.
    """
    
    opt = torch.optim.Adam(model.parameters())
    
    loss_func = torch.nn.NLLLoss() # since our model gives negative log probs on the output side
    
    loss_batch_size = 100
    
    for i in range(n_epochs):
    
        #get input stack of obs and correct labels for training
        x_train = training_data[0]
        y_train = training_data[1]
   
        pairs = list(zip(x_train, y_train))
        random.shuffle(pairs)
    
        loss = 0
        
        for x_idx, (x, y) in enumerate(pairs):
            
            if x_idx % loss_batch_size == 0:
                opt.zero_grad()
            
            x_tens = vocab.sentence_to_tensor(x, c2i)
            y_hat = model(x_tens)
            y_tens = torch.tensor(l2i[y])
       
            loss += loss_func(y_hat.unsqueeze(0), y_tens.unsqueeze(0))
            
            if x_idx % 1000 == 0:
                print(f"{x_idx}/{len(pairs)} average per-item loss: {loss / loss_batch_size}")
                
            if x_idx % loss_batch_size == 0 and x_idx > 0:
                # send back gradients:
                loss.backward()
                
                # now, tell the optimizer to update our weights:
                opt.step()
                loss = 0
        
        # now one last time:
        loss.backward()
        opt.step()
        
    return model



#analyze misses, utility function for either classifier method to visualize misses, 
#requires 2 collection.Counter objects of at least size 20, outputs bar plots to console
def analyze_class_errors(left_ends, right_ends, name):
    
    l_ends_words = []
    r_ends_words = []
    l_ends_counts = []
    r_ends_counts = []

    for k, v in left_ends.most_common(20):
        l_ends_words.append(k)
        l_ends_counts.append(v)
    

    for k, v in right_ends.most_common(20):
        r_ends_words.append(k)
        r_ends_counts.append(v)
    
    
    x_ax = np.arange(len(l_ends_words))
    fig1 = plt.figure()
    plt.bar(x_ax, l_ends_counts)
    plt.title('20 most common left_ends in missed samples')
    plt.ylabel('counts')
    plt.xticks(x_ax, l_ends_words, rotation=90)
    plt.show()
    fig1.savefig(name + 'left_ends_bar.png', bbox_inches='tight')
     
        
    fig2 = plt.figure()
    plt.bar(x_ax, r_ends_counts)
    plt.title('20 most common right_ends in missed samples')
    plt.ylabel('counts')
    plt.xticks(x_ax, r_ends_words, rotation=90)
    plt.show()
    
    fig2.savefig(name + 'right_ends_bar.png', bbox_inches='tight')
    return



