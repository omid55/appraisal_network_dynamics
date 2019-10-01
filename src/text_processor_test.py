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
