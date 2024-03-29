import math
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
            with np.errstate(divide='ignore'):
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


class SelectorBIC_orig(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    # MR: from https://discussions.udacity.com/t/bug-in-my-code-in-dic-selector-function-call-not-working-basic-python/337028/17
    # n = n_components
    # d = len(self.X[0])
    # parameters = n * n + 2 * d * n - 1
    
    these not work: @ https://github.com/ltfschoen/AIND-Recognizer/blob/master/my_model_selectors.py#L249
        param = num_state * num_state + 2 * num_state  * sum(self.lengths) - 1
        OR param = num_state * num_state + 2 * num_state  - 1   mine tried
    """
    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_model = None
        best_score = float("inf")
        for num_state in range(self.min_n_components, self.max_n_components+1):
            try:
                with np.errstate(divide='ignore'):
                    model = GaussianHMM(n_components=num_state, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                    logL = model.score(self.X, self.lengths)
                param = num_state * num_state + 2 * len(self.X[0]) - 1
                BIC_score = (-2) * logL + math.log(sum(self.lengths)) * param
                if BIC_score < best_score:
                    best_model = model
                    best_score = BIC_score
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, num_state))

        return best_model


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    # MR: from https://discussions.udacity.com/t/bug-in-my-code-in-dic-selector-function-call-not-working-basic-python/337028/17
    # n = n_components
    # d = len(self.X[0])
    # parameters = n * n + 2 * d * n - 1

    these not work: @ https://github.com/ltfschoen/AIND-Recognizer/blob/master/my_model_selectors.py#L249
        param = num_state * num_state + 2 * num_state  * sum(self.lengths) - 1
        OR param = num_state * num_state + 2 * num_state  - 1   mine tried
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_model = None
        best_score = float("inf")
        for num_state in range(self.min_n_components, self.max_n_components + 1):
            try:
                with np.errstate(divide='ignore'):
                    model = GaussianHMM(n_components=num_state, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                    logL = model.score(self.X, self.lengths)
                param = num_state * num_state + 2 * num_state * len(self.X[0]) - 1
                BIC_score = (-2) * logL + math.log(sum(self.lengths)) * param
                if BIC_score < best_score:
                    best_model = model
                    best_score = BIC_score
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, num_state))

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    
    slides @ http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_model = None
        best_score = float("-inf")
        for num_state in range(self.min_n_components, self.max_n_components+1):
            try:
                with np.errstate(divide='ignore'):
                    model = GaussianHMM(n_components=num_state, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                    logL = model.score(self.X, self.lengths)
                    DFC_score = sum([model.score(oX, olens) for ow, (oX, olens) in self.hwords.items() if ow != self.this_word])/(len(self.hwords) - 1)
                DIC_score = logL - DFC_score
                if DIC_score > best_score:
                    best_model = model
                    best_score = DIC_score
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, num_state))

        return best_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        seq_length = len(self.sequences)
        n_split = int(seq_length / 6) if seq_length > 9 else min(seq_length, 3)
        if n_split < 2:
            return None
        split_method = KFold(n_split)
        best_model = None
        best_score = float("-inf")
        for num_state in range(self.min_n_components, self.max_n_components+1):
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                try:
                    with np.errstate(divide='ignore'):
                        model = GaussianHMM(n_components=num_state, covariance_type="diag", n_iter=1000,
                                                random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                        test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)
                        logL = model.score(test_X, test_lengths)
                    if logL > best_score:
                        best_model = model
                        best_score = logL
                except:
                    if self.verbose:
                        print("failure on {} with {} states".format(self.this_word, num_states))
        return best_model
