# Omid55
# Start date:     27 Jan 2019
# Modified date:  27 Jan 2019
# Author:   Omid Askarisichani
# Email:    omid55@cs.ucsb.edu
# Processing the IM text messages module.
# Example:
#

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pandas as pd
import pickle as pk
import numpy as np
import string
import os
import re
import textblob
from xml.sax import saxutils as su

from typing import List
from typing import Tuple
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer


def get_my_stem(word: str) -> str:
    """Gets the stem of a word.

    Args:
        word: Any word in English.

    Returns:
        The stem of the word in English and if could not, just the same word.

    Raises:
        None.
    """
    p_stemmer = PorterStemmer()
    try:
        return p_stemmer.stem(word)
    except:
        return word


class EmotionDetector(object):
    def __init__(self):
        # self._anew_dicts = {}
        self._load_emotion_dictionary()

    def _load_emotion_dictionary(self):
        """Loads all of ANEW emotion words.

        Valence (ranging from pleasant to unpleasant).
        Arousal (ranging from calm to excited).
        Dominance (level of the control).

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        anew_pkl_filepath = '/home/omid/Dropbox/PhD/Projects/Appraisal Network Estimation/appraisal_network_dynamics/bagofwords/anew_dicts.pkl'
        if os.path.exists(anew_pkl_filepath):
            # loads the preprocessed anew dicts file.
            f = open(anew_pkl_filepath, 'rb')
            self._anew_dicts = pk.load(f)
        else:
            anew_dictionary_filepath = '/home/omid/Dropbox/PhD/Projects/Appraisal Network Estimation/appraisal_network_dynamics/bagofwords/ANEW_stemmed.csv'
            anew = pd.read_csv(anew_dictionary_filepath)
            self._anew_dicts = {'valence': {}, 'arousal': {}, 'dominance': {}}
            for i in range(len(anew)):
                self._anew_dicts['valence'][anew.ix[i]['Description']] = (
                    anew.ix[i]['Valence Mean'])
                self._anew_dicts['arousal'][anew.ix[i]['Description']] = (
                    anew.ix[i]['Arousal Mean'])
                self._anew_dicts['dominance'][anew.ix[i]['Description']] = (
                    anew.ix[i]['Dominance Mean'])
            with open(anew_pkl_filepath, 'wb') as handle:
                pk.dump(self._anew_dicts, handle, protocol=pk.HIGHEST_PROTOCOL)

    def compute_mean_emotion(
            self,
            content: str,
            verbose: bool = False) -> Tuple[int, float, float, float]:
        """Computes the mean emotion from the given content.

        Emotion values fall in [0, 9]: 0.0 (for words expressing no emotional
        activation), to 9.0 (for words expressing high emotional activation)
        Only a few words have emotions, which are defined based on ANEW
        dictionary. It finds those words and compute the average of their
        emotion and returns the number of words with emotion and mean emotion
        value. If no words were found, then it returns 0, 0.

        Args:
            content: String content to be used for emotion analysis.

            verbose: If we want to print words with information.

        Returns:
            Average of emotion values and the number of words with emotion.

        Raises:
            None.
        """
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(content)
        en_stop = get_stop_words('en')
        stopped_tokens = [w for w in tokens if w not in en_stop]
        stemmed_words = [get_my_stem(w) for w in stopped_tokens]
        if verbose:
            print('\n')
        count = 0
        valence_score = 0
        arousal_score = 0
        dominance_score = 0
        for word in stemmed_words:
            if word in self._anew_dicts['valence']:
                if verbose:
                    print(
                        'Word with emotion: {} = [{}, {}, {}]'.format(
                            word,
                            self._anew_dicts['valence'][word],
                            self._anew_dicts['arousal'][word],
                            self._anew_dicts['dominance'][word]))

                valence_score += self._anew_dicts['valence'][word]
                arousal_score += self._anew_dicts['arousal'][word]
                dominance_score += self._anew_dicts['dominance'][word]
                count += 1
        if count:
            valence_score /= count
            arousal_score /= count
            dominance_score /= count
        if verbose:
            print('\n')
        return count, valence_score, arousal_score, dominance_score


class SentimentAnalyzer(object):
    def __init__(self):
        self.sid = SentimentIntensityAnalyzer()

    def compute_sentiment(self, content: str) -> float:
        """Computes the sentiment (valence) of the content.

        Args:
            content: String content to be used for sentiment analysis.

        Returns:
            Sentiment of the content.

        Raises:
            None.
        """
        sentiment = self.sid.polarity_scores(content)['compound']
        return sentiment

    def compute_sentiment_with_textblob(self, content: str) -> float:
        """Computes the sentiment polarity using Textblob library.

        Args:
            content: String content to be used for sentiment analysis.

        Returns:
            Sentiment of the content.

        Raises:
            None.
        """
        blob = textblob.TextBlob(content)
        return blob.sentiment.polarity
        # for sentence in blob.sentences:
        #     print(sentence.sentiment.polarity)


# """find out if a sentence is not english"""
# def my_lang_check(word):
#     lang = enchant.Dict("en_US")
#     if word in ['lol', 'ok', 'okey', 'idk', 'sory', 'thx', 'tnx', 'tanx', 'thanx', 'hah', 'heh', 'hurtin', 'ic', 
#     'chillin', 'sux', 'nah', 'faggot', 'haha', 'nothin', 'np', 'me2', 'tv', 'lookin', 'yo', 'hmm', 'loveit', 'nutz']:
#         return True
#     return lang.check(word) or lang.check(word.capitalize())


# def is_english(sentence):
#     non_english = 0
#     lang = enchant.Dict("en_US")
#     translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
#     sentence = sentence.lower().translate(translator).strip()
#     sentence = ' '.join(sentence.split())
#     words = sentence.split()
#     digits = 0
#     for word in words:
#         if word:
#             if word.isdigit():
#                 digits += 1
#                 continue
#             if not my_lang_check(word):
#                 non_english += 1
#     if not len(words) - digits:
#         return True
#     non_english_ratio = float(non_english) / (len(words) - digits)
# #     print(non_english_ratio)
#     return non_english_ratio < 0.8