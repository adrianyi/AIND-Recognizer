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
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        bestScore = float('inf')
        # If the attempt to score fails for all cases, return the simplest case of n = min_n_components
        bestModel = self.base_model(self.min_n_components)
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n)
                logL = model.score(self.X, self.lengths)
                
                # number of features
                n_f = np.array(self.X).shape[-1]
                # number of parameters = transition matrix n*(n-1) + starting probabilities (n-1) + Gaussian parameters (2 * n_f * n)
                p = n**2 - 2 * n_f * n - 1
                # number of data points
                N = len(self.X)
                scoreNew = -2*logL + p * math.log(N)
                
                if scoreNew < bestScore:
                    score = scoreNew
                    bestModel = model
            except:
                pass
        
        return bestModel


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        # If the attempt to score fails for all cases, return the simplest case of n = min_n_components
        bestModel = self.base_model(self.min_n_components)
        bestScore = float('-inf')
        n_total = self.max_n_components + 1 - self.min_n_components
        for n in range(self.min_n_components, self.max_n_components + 1):
            # DIC(word) = logL(word) - average(logL(another_word) for another_word != word) = logL(word) - sum(logL(another_word)) / (N-1)
            # So loop through all the words, calculate logL(word) and sum(logL(!word)), then calculate DIC
            logL_word = 0
            sum_logL_notword = 0
            for word in self.words.keys():
                X_word, lengths_word in self.hwords:
                model = self.base_model(n)
                    try:
                        if word == self.this_word:
                            logL_word = model.score(X_word, lengths_word)
                        else:
                            sum_logL_notword += model.score(X_word, lengths_word)
                    except:
                        pass
            scoreNew = logL_word - sum_logL_notword / (len(self.words)-1)
            bestModel = model
        return bestModel

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        bestScore = float('-inf')
        # If the attempt to score fails for all cases, return the simplest case of n = min_n_components
        bestModel = self.base_model(self.min_n_components)
        
        n_samples = len(self.lengths)
        
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                kf = KFold(n_splits=min(3,n_samples), shuffle = True, random_state = 517)
                score_list = []
                for train_idx, test_idx in kf.split(self.sequences):
                    X_train, lengths_train = combine_sequences(train_idx, self.sequences)
                    X_test, lengths_test = combine_sequences(test_idx, self.sequences)
                    model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                    score_list.append(model.score(X_test, lengths_test))
                scoreNew = statistics.mean(score_list)
                
                if scoreNew > bestScore:
                    bestScore = scoreNew
                    bestModel = model
                    
            except:
                pass
            
        return bestModel
