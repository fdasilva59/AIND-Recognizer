import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern) 
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


    
class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
    
    def scoreBIC(self, n):
        ''' Helper function to compute and return the BIC score for a hmm with nb_states
        
            In order to compute BIC = -2 * logL + p * logN, we need to compute the number
            of free parameters p in the model. See referenecs below:
        
            http://hmmlearn.readthedocs.io/en/latest/api.html#gaussianhmm
            https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235
            https://github.com/hmmlearn/hmmlearn/blob/master/hmmlearn/hmm.py#L106
            http://hmmlearn.readthedocs.io/en/latest/tutorial.html
            ftp://metron.sta.uniroma1.it/RePEc/articoli/2002-LX-3_4-11.pdf
              
            The free parameters in the model consist in :
             - model.startprob_ : Starting probabilities (n_components array), but since 
               the probabilities add to 1.0, size (n_components-1) is used
             - model.transmat_ : Transition probabilities (n_components * n_components array),
               but since the probabilities add to 1.0, size n_components * (n_components-1) is used
             - model.means_ : Means (n_components * n_features array)
             - model.covars_ : Covariances (n_components * n_features array as covariance_type='Diag' 
               by default)
         
            Thus if n=n_components and d=n_features, p = (n-1) + (n*n(-1)) + n*d + n*d 
            which can be simplified as p = n**2 + 2*n*d -1

        '''
        
        # The hmmlearn library may not be able to train or score all models but 
        # in case of error, the exception will be managed by the caller function
        
        # Retrieve the Hidden Markov Model with Gaussian emissions (with n states)
        hmm_model = self.base_model(n)

        # Compute the Bayesian information criteria
        logL = hmm_model.score(self.X, self.lengths)                      
        d = hmm_model.n_features
        p = n**2 + 2*n*d -1    
        logN = np.log(sum(self.lengths))
        bic = -2 * logL + p * logN

        # Return the BIC score
        # Note : we choose to not return the hmm_model to avoid troubles with objects reference 
        # over the different calls/iterations on this helper function. The select function (caller) will
        # recompute a model, using the number of states which will have returned the best BIC score value
        if self.verbose:
            print("[Debug SelectorBIC:score_bic] nb_states=", n, " BIC=", bic) 
        return bic


    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        #raise NotImplementedError
        
        best_n_components = None
        best_bic = math.inf
        
        for n in range(self.min_n_components, self.max_n_components+1):
            
            try:
                # Compute the Bayesian Information Criterion for a hmm with n states                
                bic_score = self.scoreBIC(n) 

                # Track the number of hhm states that returns the best BIC score
                if bic_score < best_bic :
                    best_bic = bic_score
                    best_n_components = n
                    if self.verbose:
                        print("[Debug SelectorBIC:select] new best_bic=", best_bic , " for n=",n, " states")
            except:
                # The hmmlearn library may not be able to train or score all models :
                # in case of error, just move to the next iteration
                if self.verbose:
                    print("[Debug SelectorBIC:select] Exception caught with ",n, " hidden states")
                pass
                
        # Finally return the model with number of hidden states that produced the best BIC score        
        return self.base_model(best_n_components)


    
class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def score_dic(self, nb_states):
        ''' Helper function to compute and return the DIC score for a hmm with nb_states

        '''
            
        # The hmmlearn library may not be able to train or score all models but 
        # in case of error, the exception will be managed by the caller function
            
        # Retrieve the Hidden Markov Model with Gaussian emissions (with nb_states states)
        hmm_model = self.base_model(nb_states)

        # Compute the Log Likelihood for the hmm model with nb_states hidden states
        # This is the term log(P(X(i)) in the DIC formula
        logL = hmm_model.score(self.X, self.lengths)   

        # Data preparation : compute a copy of hwords dict without this_word 
        # This is the X(all but i) term in the Formula
        hwords_without_i = dict(self.hwords)
        hwords_without_i.pop(self.this_word)
        if self.verbose:
            print("[Debug SelectorDIC:score_dic] len(self.hwords)=",len(self.hwords), " len(hwords_without_i)=",len(hwords_without_i))
        
        # Compute the average anti-likelihood or the hmm model with nb_states hidden states
        # This is the 1/(M-1)SUM(log(P(X(all but i)) term in the DIC formula
        antiLogL_scores = []
        avg_antiLogL=0
        for w, (x,l) in hwords_without_i.items() :
            try:
                antiLogL_scores.append(hmm_model.score(x, l)) 
            except:
                # The hmmlearn library may not be able to train or score all models :
                # in case of error, just move to the next iteration
                if self.verbose:
                    print("[Debug SelectorDIC:score_dic] WARNING : One exception ignored when computing anti-likelihood for ", self.this_word)
                pass
                                       
        if len(antiLogL_scores)<1:
            avg_antiLogL=0
            if self.verbose:
                print("[Debug SelectorDIC:score_dic] WARNING : antiLogL_scores empty")
        else:
            avg_antiLogL = np.mean(antiLogL_scores)
        
        # Return the DIC score
        # Note : we choose to not return the hmm_model to avoid troubles with objects reference 
        # over the different calls/iterations on this helper function. The select function (caller) 
        # will recompute a model with the full dataset, using the number of states 
        # which will have returned the best mean DIC score value
        
        #if self.verbose:
        #print("[Debug SelectorDIC:score_dic] nb_states=", nb_states, " LogL=", logL, " avg_antiLogL=", avg_antiLogL, " DIC=", (logL - avg_antiLogL)) 
        return (logL - avg_antiLogL)
    

                                   
                                
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        #raise NotImplementedError
        
        best_n_components = None
        best_dic = -math.inf
        
        for n in range(self.min_n_components, self.max_n_components+1):
        
            try:
                # Compute the Discriminative Information Criterion for a hmm with n states                
                dic_score = self.score_dic(n) 

                # Track the number of hhm states that returns the best DIC score
                if dic_score > best_dic:
                    best_dic = dic_score
                    best_n_components = n
                    #if self.verbose:
                    #print("[Debug SelectorDIC:select] new best_dic=", best_dic , " for n=",n," states")
                                   
            except:
                # The hmmlearn library may not be able to train or score all models :
                # in case of error, just move to the next iteration
                if self.verbose:
                    print("[Debug SelectorDIC:select] Exception caught with ",n, " hidden states")
                pass
        
        # Return the hmm_model using the best number of components/hidden states configuration
        return self.base_model(best_n_components)                            



class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def score_cv(self, nb_states, kf_n_splits): 
        ''' Helper function to compute and return the average log Likelihood of 
            cross-validation folds for a hmm with nb_states

        '''
        
        # The hmmlearn library may not be able to train or score all models but 
        # in case of error, the exception will be managed by the caller function
        
        if self.verbose:
            print("[Debug SelectorCV:score_cv] nb_states=", nb_states)
    
        # Initialize K-Fold (See http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
        # Using kf_n_splits folds should be enough to produce good results while not penalizing too much the execution time
        split_method = KFold(n_splits=kf_n_splits, shuffle=False, random_state=self.random_state)
        
        # Initialize an array to store the LogL values of each folds
        LogL_scores = []
       
        # Train and compute the K-Folds cross validation         
        for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
            
            if self.verbose:
                print("[Debug SelectorCV:score_cv] Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx))

            # Training subset : Combine subsets of X and lengths tuples for the training fold
            # (base_model() is fitting the model with the training set provided in self.X, self.lengths)
            self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)

            # Test subset : Combine subsets of X and lengths tuples for the cross validation fold
            X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)

            # Train and Retrieve the Hidden Markov Model with Gaussian emissions (with nb_states states)
            hmm_model = self.base_model(nb_states)

            # Compute the Log Likelihood score on the cross validation fold (using the HMM model with nb_states States)
            LogL_scores.append(hmm_model.score(X_test, lengths_test))

            if self.verbose:
                print("[Debug SelectorCV:score_cv] LogL_scores=", LogL_scores)     
 
        if len(LogL_scores)<1:
            if self.verbose:
                print("[Debug SelectorCV:score_cv] LogL_scores emty") 
            return -math.inf
             
        if self.verbose:
            print("[Debug SelectorCV:score_cv] nb_states=", nb_states, " average LogL_scores=", np.mean(LogL_scores))
        
        # Note : we choose to not return the hmm_model because it would correspond to the last K-Fold iteration
        # The select function (caller) will recompute a model with the full dataset, using the number of states 
        # which will have returned the best mean score value        
        return np.mean(LogL_scores)
    
    
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        #raise NotImplementedError
         
        best_n_components = None
        best_score = -math.inf  
        kf_n_splits = 3
        
        # Check if there are  enough sequences to perform K-Fold with n_splits 
        if len(self.sequences) < kf_n_splits:
            if self.verbose:
                print("[Debug SelectorCV:score_cv] Not enough sequences to perform k-folds CV for : ", self.this_word, "len(self.sequences)=",len(self.sequences), "K-Fold CV n_splits=", kf_n_splits)
            if len(self.sequences)>=2:
                kf_n_splits = len(self.sequences)
                if self.verbose:
                    print("[Debug SelectorCV:score_cv] Constraining K-Fold CV with n_splits=", kf_n_splits, " for ", self.this_word)
            else:
                # K-Fold n_splits cannot be less than 2
                return None
       
        for n in range(self.min_n_components, self.max_n_components+1):
            
            try:
            
                # Compute the average log Likelihood of cross-validation folds for a hmm with n states
                logL=self.score_cv(n, kf_n_splits) 

                # Track the number of hhm states that returns the best LogL score
                if logL > best_score:
                    best_score = logL
                    best_n_components = n
                    if self.verbose:
                        print("[Debug SelectorCV:select] NEW best_score=", best_score , " for n=",n, " states")
                
            except:
                if self.verbose:
                    print("[Debug SelectorCV:select] Exception caught with ",n, " hidden states")
                pass                
                
        # restore X and lengths variables to their initial values, as they have been modified during the 
        # K-Fold  cross-validation process
        self.X, self.lengths = self.hwords[self.this_word]
        
        # Return the hmm_model, using the whole dataset and the best number of components/hidden states configuration
        return self.base_model(best_n_components)

        
        
