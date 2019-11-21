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
import csv
import os
import re
import textblob
from xml.sax import saxutils as su
from os.path import expanduser

from typing import List
from typing import Tuple
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer

import pogs_jeopardy_log_lib as lib


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


class SlangToFormalTranslator(object):
    def __init__(self, messages):
        self._load_slang_file()
        self.messages = messages

    def _load_slang_file(self):
        slang_file = open("slang.txt", "r")
        self.slang_dict = {}
        count = 0
        for row in slang_file:
            count += 1
            strings = row.split("=")
            self.slang_dict[str(strings[0])] = str(
                strings[1].replace("\n", ""))
        slang_file.close()

    def _translate_string(self, event_content):
        event_content_array = event_content.split(" ")
        j = 0
        for word in event_content_array:
            if word.upper() in self.slang_dict.keys():
                event_content_array[j] = self.slang_dict[word.upper()]
            j += 1
        return " ".join(event_content_array)

    def _translate_messages(self):
        for i in range(len(self.messages)):
            self.messages[i].event_content = self.messages[i].event_content.apply(
                self._translate_string)


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
        #anew_pkl_filepath = expanduser('~/Dropbox/PhD/Projects/Appraisal Network Estimation/appraisal_network_dynamics/bagofwords/anew_dicts.pkl')
        anew_pkl_filepath = expanduser(
            '/home/koasato/Documents/research/appraisal_network_dynamics/bagofwords/anew_dicts.pkl')
        if os.path.exists(anew_pkl_filepath):
            # loads the preprocessed anew dicts file.
            f = open(anew_pkl_filepath, 'rb')
            self._anew_dicts = pk.load(f)
        else:
            #anew_dictionary_filepath = expanduser('~/Dropbox/PhD/Projects/Appraisal Network Estimation/appraisal_network_dynamics/bagofwords/ANEW_stemmed.csv')
            anew_dictionary_filepath = expanduser(
                '/home/koasato/Documents/research/appraisal_network_dynamics/bagofwords/ANEW_stemmed.csv')
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


class TextPreprocessor(object):
    def __init__(self):
        self._load_messages_for_team()

    def _load_messages_for_team(self, id=10):
        """Loads the messages for a specific team

        Args:
            id: the team_id of the team to load the logs for.

        Returns:
            None.

        Raises:
            None.
        """
        team_log_processor = lib.TeamLogProcessor(team_id=id,
                                                  logs_directory_path='/home/koasato/Documents/research/Jeopardy/')
        self.messages = team_log_processor.messages

    def _get_messages_as_list(self):
        """Aggregates all the messages in a sequential list from the Dataframe

        Args:
            None.

        Returns:
            List of messages as strings

        Raises:
            None.
        """
        messages = []
        for index in range(len(self.messages)):
            messages_per_question = list(
                self.messages[index].event_content.values[:])
            for message in messages_per_question:
                messages.append(message)
        return messages

    def _compute_emotion(self):
        """Computes emotion values for all messages

        Args:
            None.

        Returns:
            Average of emotion values and the number of words with emotion.

        Raises:
            None.
        """
        emotion_detector = EmotionDetector()
        results = []
        for message in self.message_list:
            results.append(emotion_detector.compute_mean_emotion(message))

        return results

    def _compute_sentiment(self):
        """Computes emotion values for all messages

        Args:
            None.

        Returns:
            Average of emotion values and the number of words with emotion.

        Raises:
            None.
        """
        sentiment_analyzer = SentimentAnalyzer()
        results = []
        text_blob_results = []

        for message in self.message_list:
            results.append(sentiment_analyzer.compute_sentiment(message))
            text_blob_results.append(
                sentiment_analyzer.compute_sentiment_with_textblob(message))

        for i in range(len(self.message_list)):
            print(self.message_list[i] + "  " + str(text_blob_results[i]))
        return results, text_blob_results

    def _preprocess(self, translate_slang):
        """Preprocesses the message data for a team

        Args:
            translate_slang: If the messages should substitute acronyms for
            formal English (e.g. LOL -> Laughing out Loud).

        Returns:
            None.

        Raises:
            None.
        """
        if translate_slang:
            self.slang_translator = SlangToFormalTranslator(self.messages)
            self.slang_translator._translate_messages()
        self.message_list = self._get_messages_as_list()

        self.emotion_results = self._compute_emotion()
        self.sentiment_results, self.sentiment_text_blob_results = self._compute_sentiment()


tp = TextPreprocessor()
tp._preprocess(translate_slang=True)


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
