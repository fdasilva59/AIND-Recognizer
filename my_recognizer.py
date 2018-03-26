import warnings
from asl_data import SinglesData

import math


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    
    # TODO implement the recognizer
    # return probabilities, guesses
    #raise NotImplementedError

    
    # Loop over the entire test_set
    for word_sequences in test_set.get_all_sequences():
        
        # Intialize an empty dict to store the computed score of the  
        # word sequences for each model, keyed by its corresponding key
        scores = {}
        
        # Intialize variables to kKeep track of the best score and corresponding word
        best_score = -math.inf
        word_guess = None
        
        # Retrive the sequence list for each word item in the test_set
        X, lengths = test_set.get_item_Xlengths(word_sequences)
            
        # Iterate over the dictionary of trained models keyed by word    
        for word, model in models.items(): 
            
            try:
                # Score the sequences against the current model iteration
                scores[word] = model.score(X, lengths)
                
                # Keep track of the best score and corresponding word
                if best_score < scores[word]:
                    best_score = scores[word] 
                    word_guess=word
                    
            except:
                
                # The hmmlearn library may not be able to score all models : 
                # in case of error, we move to the next iteration
                scores[word] = None
                pass

        # Add the scores dictionnary for the corresponding word in the probability list
        probabilities.append(scores)
        # and happen the guesses word to the guesses list
        guesses.append(word_guess)

    return probabilities, guesses
