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
                    all_score[word] = float("-inf")
        probabilities.append(all_score)
        df_prob = pd.DataFrame(probabilities)
        df_prob = df_prob.div(df_prob.sum(axis=1), axis=0)
    guesses = [builtins.sorted(sd.items(), key=lambda x: x[1], reverse=True)[0][0] for sd in probabilities]

    return probabilities, guesses

#  https://stackoverflow.com/questions/14617601/implementing-ngrams-in-python
def unigrams(text):
    uni = []
    for token in text:
        uni.append([token])
    return uni

def bigrams(text):
    bi = []
    token_address = 0
    for token in text[:len(text) - 1]:
        bi.append([token, text[token_address + 1]])
        token_address += 1
    return bi

def trigrams(text):
    tri = []
    token_address = 0
    for token in text[:len(text) - 2]:
        tri.append([token, text[token_address + 1], text[token_address + 2]])
        token_address += 1
    return tri

def ngram(text, grams):
    model=[]
    count=0
    for token in text[:len(x)-grams+1]:
       model.append(text[count:count+grams])
       count=count+1
    return model


''' https://web.stanford.edu/class/cs124/lec/languagemodeling.pdf
Reminder:The Chain Rule
• Recall the definiton of condional probabilies
• More variables:
   P(A,B,C,D) = P(A)P(B|A)P(C|A,B)P(D|A,B,C)
• The Chain Rule in General
   P(x1,x2,x3,…,xn) = P(x1)P(x2|x1)P(x3|x1,x2)…P(xn|x1,…,xn,1)

N-gram : https://en.wikipedia.org/wiki/N-gram
   https://stackoverflow.com/questions/14617601/implementing-ngrams-in-python
   https://stackoverflow.com/questions/13423919/computing-n-grams-using-python

Selectors: http://avansp.github.io/2014/11/02/DIC-AIC-BIC.html
   http://www.petrkeil.com/?p=836
   https://arxiv.org/pdf/1307.5928.pdf
   http://www.math.uni.wroc.pl/~mbogdan/Preprints/Dutta.pdf
   http://www.columbia.edu/~jwp2128/Teaching/W4721/Spring2017/slides/lecture_4-25-17.pdf
   http://www.sciencedirect.com/science/article/pii/S1877050916315198

'''