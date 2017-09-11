import builtins
import warnings
import numpy as np
from asl_data import SinglesData


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

    wid_list = builtins.sorted(test_set.get_all_sequences().keys())
    for wid in wid_list:
        X, lengths = test_set.get_item_Xlengths(wid)
        all_score = {}
        for word, model in models.items():
            if model:
                try:
                    with np.errstate(divide='ignore'):
                        logL = model.score(X, lengths)
                    all_score[word] = logL
                except:
                    pass
        probabilities.append(all_score)
    guesses = [builtins.sorted(sd.items(),key=lambda x: x[1], reverse=True)[0][0] for sd in probabilities]
    return probabilities, guesses


