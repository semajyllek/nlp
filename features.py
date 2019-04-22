from  hw2_utils import constants 
import numpy as np

# deliverable 4.1
def get_token_type_ratio(counts):
    """
    Compute the ratio of tokens to types
    
    :param counts: bag of words feature for a song
    :returns: ratio of tokens to types
    :rtype float
    """
        
    return sum(counts.values()) / len(counts) if len(counts) > 0 else 0
    

# deliverable 4.2
def concat_ttr_binned_features(data):
    """
    Add binned token-type ratio features to the observation represented by data
    
    :param data: Bag of words
    :returns: Bag of words, plus binned ttr features
    :rtype: dict
    """
    
    ttr = get_token_type_ratio(data)
    
    data[constants.TTR_ZERO] = 0
    data[constants.TTR_ONE] = 0
    data[constants.TTR_TWO] = 0
    data[constants.TTR_THREE] = 0
    data[constants.TTR_FOUR] = 0
    data[constants.TTR_FIVE] = 0
    data[constants.TTR_SIX] = 0
    

    #lazy python novice stuff
    if ttr < 1:
        data[constants.TTR_ZERO] = 1
        return data
    elif ttr < 2:
        data[constants.TTR_ONE] = 1
        return data
    elif ttr < 3:
        data[constants.TTR_TWO] = 1
        return data
    elif ttr < 4:
        data[constants.TTR_THREE] = 1
        return data
    elif ttr < 5:
        data[constants.TTR_FOUR] = 1
        return data
    elif ttr < 6:
        data[constants.TTR_FIVE] = 1
        return data
    else:
        data[constants.TTR_SIX] = 1
        return data
  
