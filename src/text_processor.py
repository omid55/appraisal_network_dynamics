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

import csv
import enchant
import numpy as np
import os
import pandas as pd
import pickle as pk
import re
import string
import textblob
from xml.sax import saxutils as su
from os.path import expanduser

from typing import List
from typing import Text
from typing import Tuple
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TweetTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer

import pogs_jeopardy_log_lib as lib
import utils


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
        anew_pkl_filepath = expanduser(
            '~/Dropbox/PhD/Projects/Appraisal Network Estimation/appraisal_network_dynamics/bagofwords/anew_dicts.pkl')
        # anew_pkl_filepath = expanduser(
        #     '~/Documents/koa/College/UCSB/2019-2020/Research/appraisal_network_dynamics/bagofwords/anew_dicts.pkl')
        if os.path.exists(anew_pkl_filepath):
            # loads the preprocessed anew dicts file.
            f = open(anew_pkl_filepath, 'rb')
            self._anew_dicts = pk.load(f)
        else:
            anew_dictionary_filepath = expanduser(
                '~/Dropbox/PhD/Projects/Appraisal Network Estimation/appraisal_network_dynamics/bagofwords/ANEW_stemmed.csv')
            # anew_dictionary_filepath = expanduser(
            #     '~/Documents/koa/College/UCSB/2019-2020/Research/appraisal_network_dynamics/bagofwords/ANEW_stemmed.csv')
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
    def _get_messages_as_sequential_list(self):
        """Aggregates all the messages in a sequential list in the form:
        [ user_id, message ]

        Args:
            None.

        Returns:
            List of messages where each message is a list as described above.

        Raises:
            None.
        """
        messages = []
        for index in range(len(self.messages)):
            messages_of_question = list(
                self.messages[index].content.values[:])
            senders_of_message_per_question = list(
                self.messages[index].sender_subject_id.values[:])
            for i in range(len(messages_of_question)):
                messages.append(
                    [senders_of_message_per_question[i],
                     messages_of_question[i]])
        return messages

    def _get_messages_as_aggregated_dict(self):
        """Aggregates all the messages in an aggregated list per user for every 
        5 questions, all naively concatenated with " / "

        Args:
            None.

        Returns:
            Dict of aggregated messages for every 5 questions

        Raises:
            None.
        """
        aggregated_messages = {}
        message_group = {}
        for index in range(len(self.messages)):
            if ((index // 5) in message_group.keys()):
                message_group[index // 5] = message_group[
                    index // 5].append(self.messages[index])
            else:
                message_group[index // 5] = self.messages[index]

        for index in range(len(message_group)):
            message_dict = {}
            for id in self.team_members:
                message_dict[id] = ""

            for message_index, row in message_group[index].iterrows():
                message_dict[int(row.sender_subject_id)
                             ] = message_dict[int(row.sender_subject_id)
                                              ] + " / " + row.content
            aggregated_messages[index] = message_dict
        return aggregated_messages

    def _compute_emotion(self, messages):
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
        if (isinstance(messages, list)):
            for message in messages:
                results.append(
                    [message[0], emotion_detector.compute_mean_emotion(message[1])])
        elif (isinstance(messages, dict)):
            for key in messages.keys():
                for subkey in messages[key]:
                    results.append(
                        [subkey, emotion_detector.compute_mean_emotion(messages[key][subkey])])

        return results

    def _compute_sentiment(self, messages):
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

        if (isinstance(messages, list)):
            for message in messages:
                results.append(
                    [message[0], sentiment_analyzer.compute_sentiment(message[1])])
                text_blob_results.append(
                    [message[0], sentiment_analyzer.compute_sentiment_with_textblob(
                        message[1])])
        elif (isinstance(messages, dict)):
            for key in messages.keys():
                for subkey in messages[key]:
                    results.append(
                        [subkey, sentiment_analyzer.compute_sentiment(
                            messages[key][subkey])])
                    text_blob_results.append(
                        [subkey, sentiment_analyzer.compute_sentiment_with_textblob(
                            messages[key][subkey])])

        return results, text_blob_results

    def _preprocess(self, translate_slang, team_id):
        """Preprocesses the message data for a team

        Args:
            translate_slang: If the messages should substitute acronyms for
            formal English (e.g. LOL -> Laughing out Loud).

        Returns:
            None.

        Raises:
            None.
        """

        self._load_messages_for_team(
            team_id,
            logs_directory_path=expanduser(
                '~/Datasets/Jeopardy/'))
                # '~/Documents/koa/College/UCSB/2019-2020/Research/Jeopardy/'))
        if translate_slang:
            self.slang_translator = SlangToFormalTranslator()
            self.messages = self.slang_translator.translate_messages(
                self.messages)
        message_list = self._get_messages_as_sequential_list()
        aggregated_message_list = self._get_messages_as_aggregated_dict()

        self.emotion_results = self._compute_emotion(message_list)
        self.sentiment_results, self.sentiment_text_blob_results = self._compute_sentiment(
            message_list)

        self.aggregated_emotion_results = self._compute_emotion(
            aggregated_message_list)
        self.aggregatedsentiment_results, self.sentiment_text_blob_results = self._compute_sentiment(
            aggregated_message_list)


class FormalEnglishTranslator(object):
    """Changes slang or wrong text to formal and correct English.

        Usage:
            fixer = FormalEnglishTranslator()
            dataframe = fixer.translate_messages(dataframe)

        Properties:
            slang_dict: Maps slang text to formal English sentences.
    """
    def __init__(
        self, slang_dictionary_filepath: Text = 'bagofwords/slang.txt'):
        self._load_slang_dictionary(
            slang_dictionary_filepath=slang_dictionary_filepath)
        self._lang = enchant.Dict("en_US")

    def _load_slang_dictionary(
        self, slang_dictionary_filepath: Text) -> None:
        self.slang_dict = {}
        with open(slang_dictionary_filepath, 'r') as slang_file:    
            count = 0
            for row in slang_file:
                count += 1
                strings = row.split('=')
                self.slang_dict[str(strings[0]).lower()] = str(
                    strings[1].replace('\n', '')).lower()

    def _get_formal_text(self,
                         content: Text,
                         fix_spelling: bool = False) -> Text:
        """Gets the formal and correct version of the input text.
        
        Args:
            content: Content of message with slang and wrong English.

            fix_spelling: Whether to fix spelling with the closest English word.

        Returns:
            Content of the message after fixing slang or wrong text.

        Raises:
            None.
        """
        # if not slang_dict:
        #     self._load_slang_dictionary()
        tokenizer = TweetTokenizer()
        tokens = tokenizer.tokenize(content.lower())
        for j, word in enumerate(tokens):
            if word in self.slang_dict.keys():
                tokens[j] = self.slang_dict[word]
        formal_content = ' '.join(tokens)
        if fix_spelling:
            tokens = tokenizer.tokenize(formal_content)
            for j, token in enumerate(tokens):
                # has_any_letter = bool(re.search('[a-zA-Z]', token))
                has_any_letter = bool(re.search('[a-z]', token))
                if not self._lang.check(token) and has_any_letter:
                    tokens[j] = self._lang.suggest(token)[0]
            formal_content = ' '.join(tokens)
        return formal_content

    def translate_messages(self,
                           messages: pd.DataFrame,
                           fix_spelling: bool = False,
                           message_column_name: Text = 'event_content',
                           ) -> pd.DataFrame:
        """Translates ...
        
        Args:

        Returns:

        Raises:
            ValueError: If the message column name does not exist.
        """
        utils.check_required_columns(messages, [message_column_name])
        new_messages = messages.copy()
        new_messages[message_column_name] = messages[message_column_name].apply(
            self._get_formal_text, fix_spelling=fix_spelling)
        return new_messages
