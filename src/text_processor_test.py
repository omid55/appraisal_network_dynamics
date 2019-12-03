# Omid55
# Test module for text_processor.

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import pandas as pd
import pandas.testing as pd_testing
import unittest
import os
from os.path import expanduser
from parameterized import parameterized

import text_processor
import pogs_jeopardy_log_lib as lib


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
# *********************** EmotionDetector class *******************************
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
# *********************** FormalEnglishTranslator class ***********************
# *****************************************************************************
class FormalEnglishTranslatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.translator = text_processor.FormalEnglishTranslator()

    # =========================================================================
    # ===================== _load_slang_dictionary ============================
    # =========================================================================
    def test_load_slang_dictionary(self):
        self.assertEqual(
            self.translator._slang_dict['afaik'], 'as far as i know')
        self.assertEqual(self.translator._slang_dict['b4n'], 'bye for now')
        self.assertEqual(self.translator._slang_dict['gn'], 'good night')
        self.assertEqual(self.translator._slang_dict['ltns'], 'long time no see')
        self.assertEqual(self.translator._slang_dict['7k'], 'sick:-d laughter')

    # =========================================================================
    # =========================== _get_formal_text ============================
    # =========================================================================
    @parameterized.expand([
        ['sup bro?', 'what is up brother ?', False],
        ['sup frined?', 'what is up friend ?', True]
        ])
    def test_get_formal_text(self, content, expected, fix_spelling):
        content = 'sup bro?'
        expected = 'what is up brother ?'
        computed = self.translator._get_formal_text(
            content=content, fix_spelling=False)
        self.assertEqual(expected, computed)

    # =========================================================================
    # ========================= translate_messages ============================
    # =========================================================================
    def test_translate_messages_raises_when_not_existing_column(self):
        data = pd.DataFrame({
            'message_content': [
                'sup man?',
                'hello my freind'],
            'sender_subject_id': [123, 122]})
        with self.assertRaises(ValueError):
            self.translator.translate_messages(
                messages=data,
                message_column_name='not_existing_column_name')

    def test_translate_messages(self):
        data = pd.DataFrame({
            'message_content': [
                'sup man?',
                'hello my freind',
                'what\'s up buddy?',
                'nothing',
                'wut wsa that!!'],
            'sender_subject_id': [123, 122, 123, 122, 124]})
        expected = pd.DataFrame({
            'message_content': [
                'what is up man ?',
                'hello my friend',
                'what\'s up buddy ?',
                'nothing',
                'what was that ! !'],
            'sender_subject_id': [123, 122, 123, 122, 124]})
        computed = self.translator.translate_messages(
            messages=data,
            fix_spelling=True,
            message_column_name='message_content')
        pd_testing.assert_frame_equal(expected, computed)
