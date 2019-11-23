# Omid55
# Test module for text_processor.

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import pandas as pd
import unittest
import os
from parameterized import parameterized

import text_processor


# =========================================================================
# ==================== has_numbers ========================================
# =========================================================================
@parameterized.expand(
    [["hi there5 no!?", True],
        ["hi thereooo o no!?", False]])
def test_has_numbers(self, text, expected):
    computed = text_processor.has_numbers(text)
    self.assertEqual(expected, computed)

# *****************************************************************************
# *********************** EmotionDetector class *****************************
# *****************************************************************************


class EmotionDetectorTest(unittest.TestCase):
    # =========================================================================
    # =========================== compute_mean_emotion ========================
    # =========================================================================
    @parameterized.expand([
        ['p1', "hello my friend, how was your day?", (2, 7.2, 5.37, 6.31)],
        ['p2', "you go to hell", (1, 2.24, 5.38, 3.24)],
        ['p3', "there are two stocks in my portfolio today.", (0, 0, 0, 0)]])
    def test_compute_mean_emotion(self, name, text, emotion):
        ed = text_processor.EmotionDetector()
        expected = emotion
        computed = ed.compute_mean_emotion(text)  # verbose=True
        np.testing.assert_array_almost_equal(expected, computed)


# *****************************************************************************
# *********************** SentimentAnalyzer class *****************************
# *****************************************************************************
class SentimentAnalyzerTest(unittest.TestCase):
    # =========================================================================
    # =========================== compute_sentiment ===========================
    # =========================================================================
    @parameterized.expand([
        ["hello my friend, how was your day?", 0.4939],
        ["go to hell, yo", -0.6808],
        ["there are two stocks in my portfolio today.", 0]])
    def test_compute_sentiment(self, text, sentiment):
        sa = text_processor.SentimentAnalyzer()
        expected = sentiment
        computed = sa.compute_sentiment(text)
        self.assertEqual(expected, computed)

    # =========================================================================
    # =========================== compute_sentiment ===========================
    # =========================================================================
    @parameterized.expand([
        ["hello my friend, how was your day?", 0.0],
        ["go to hell", 0.0],
        ["Best wishes for you buddy", 1.0]])
    def test_compute_sentiment_with_textblob(self, text, sentiment):
        sa = text_processor.SentimentAnalyzer()
        expected = sentiment
        computed = sa.compute_sentiment_with_textblob(text)
        self.assertEqual(expected, computed)

# *****************************************************************************
# *********************** SlangToFormalTranslator class ***********************
# *****************************************************************************


class SlangToFormalTranslatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.text_preprocessor = text_processor.TextPreprocessor()
        self.text_preprocessor._load_messages_for_team(
            id=10, logs_directory_path='/home/koasato/Documents/research/Jeopardy/')
        self.translator = text_processor.SlangToFormalTranslator(
            self.text_preprocessor.messages)

    # =========================================================================
    # =========================== _load_slang_file ============================
    # =========================================================================
    def test_slang_file_contents_length(self):
        self.assertEqual(len(self.translator.slang_dict), 73)

    def test_slang_file_contents(self):
        self.assertEqual(
            self.translator.slang_dict['AFAIK'], 'As Far As I Know')
        self.assertEqual(self.translator.slang_dict['B4N'], 'Bye For Now')
        self.assertEqual(self.translator.slang_dict['GN'], 'Good Night')
        self.assertEqual(
            self.translator.slang_dict['LTNS'], 'Long Time No See')
        self.assertEqual(self.translator.slang_dict['7K'], 'Sick:-D Laughter')

    # =========================================================================
    # =========================== _translate_string ===========================
    # =========================================================================
    def test_translate_messages(self):
        correct_translated_messages_for_question_0 = [
            'radio i think', 'yea radio seems like the move']
        correct_translated_messages_for_question_2 = [
            "I'm like 65% sure its not lincoln",
            "no clue",
            "also cant see others answers",
            "I Don't Know",
            "yes",
            "i put taft",
            "anyone know what year taft was pres",
            "for no reason in particular"]
        correct_translated_messages_for_question_27 = []
        correct_translated_messages_for_question_43 = [
            'kinda guessed',
            'Its for sure not britain',
            "my gut says iceland'",
            'i feel like its iceland Because i think france currently doesnt have a true democracy no?',
            'I Agree']

        self.translator._translate_messages()

        self.assertListEqual(correct_translated_messages_for_question_0,
                             list(self.translator.messages[0].event_content.values[:]))
        self.assertListEqual(correct_translated_messages_for_question_2,
                             list(self.translator.messages[2].event_content.values[:]))
        self.assertListEqual(correct_translated_messages_for_question_27,
                             list(self.translator.messages[27].event_content.values[:]))
        self.assertListEqual(correct_translated_messages_for_question_43,
                             list(self.translator.messages[43].event_content.values[:]))
