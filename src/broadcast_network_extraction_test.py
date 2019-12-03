# Omid55
# Test module for broadcast_network_extraction.

from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
import pandas as pd
import networkx as nx
import unittest
import utils

import broadcast_network_extraction as bne


class AggregationTypeTest(unittest.TestCase):
    # =========================================================================
    # ============================ get_function ===============================
    # =========================================================================
    def test_get_function_binary(self):
        aggregator = bne.AggregationType.BINARY
        func = aggregator.get_function()
        self.assertEqual(1, func([9, 2]))

    def test_get_function_binary_when_nothing(self):
        aggregator = bne.AggregationType.BINARY
        func = aggregator.get_function()
        self.assertEqual(0, func([]))

    def test_get_function_sum(self):
        aggregator = bne.AggregationType.SUM
        func = aggregator.get_function()
        self.assertEqual(12, func([1, 5, 6]))

    def test_get_function_average(self):
        aggregator = bne.AggregationType.AVERAGE
        func = aggregator.get_function()
        self.assertEqual(5, func([8, 1, 5, 6]))

    def test_get_function_last(self):
        aggregator = bne.AggregationType.LAST
        func = aggregator.get_function()
        self.assertEqual(6, func([1, 55, 6]))

    def test_get_function_max(self):
        aggregator = bne.AggregationType.MAX
        func = aggregator.get_function()
        self.assertEqual(6, func([1, 5, 6]))

    def test_get_function_min(self):
        aggregator = bne.AggregationType.MIN
        func = aggregator.get_function()
        self.assertEqual(1, func([1, 5, 6]))

    def test_whether_get_function_last_raises_when_nothing(self):
        with self.assertRaises(ValueError):
            aggregator = bne.AggregationType.LAST
            func = aggregator.get_function()
            func([])


class ApplyFunctionOnGraphTest(unittest.TestCase):
    # =========================================================================
    # ================= apply_function_on_all_edges ===========================
    # =========================================================================
    def test_apply_function_on_all_edges(self):
        dg = nx.DiGraph()
        dg.add_nodes_from([122, 123, 127])
        dg.add_edge(122, 123, weight=[6.3, 9.2, 2.5])
        dg.add_edge(123, 127, weight=[2.9])
        extractor = bne.NetworkExtraction()
        computed_graph = extractor._apply_function_on_all_edges(
            dgraph=dg, aggregation_type=bne.AggregationType.AVERAGE)
        expected_graph = nx.DiGraph()
        expected_graph.add_nodes_from([122, 123, 127])
        expected_graph.add_edge(122, 123, weight=6.0)
        expected_graph.add_edge(123, 127, weight=2.9)
        utils.assert_graph_equals(computed_graph, expected_graph)


class ComputeWeight(unittest.TestCase):
    # =========================================================================
    # ================= extract_network_from_broadcast ========================
    # =========================================================================
    @classmethod
    def setUpClass(cls):
        cls.extractor = bne.NetworkExtraction()

    def test_compute_weight_just_count(self):
        computed_weight = self.extractor._compute_weight(
            time1=pd.Timestamp(
                year=2019, month=1, day=1, hour=9, minute=0, second=0),
            time2=pd.Timestamp(
                year=2019, month=1, day=1, hour=9, minute=0, second=20),
            weight_type=bne.WeightType.NONE)
        expected_weight = 1
        self.assertEqual(expected_weight, computed_weight)

    def test_compute_weight_duration(self):
        computed_weight = self.extractor._compute_weight(
            time1=pd.Timestamp(
                year=2019, month=1, day=1, hour=9, minute=0, second=0),
            time2=pd.Timestamp(
                year=2019, month=1, day=1, hour=9, minute=0, second=10),
            weight_type=bne.WeightType.REPLY_DURATION,
            gamma=0.1)
        expected_weight = np.exp(-1)
        self.assertEqual(expected_weight, computed_weight)

    def test_compute_weight_sentiment(self):
        computed_weight = self.extractor._compute_weight(
            time1=pd.Timestamp(
                year=2019, month=1, day=1, hour=9, minute=0, second=0),
            time2=pd.Timestamp(
                year=2019, month=1, day=1, hour=9, minute=0, second=10),
            weight_type=bne.WeightType.SENTIMENT,
            content='hello my friend. How has your day been?')
        expected_weight = 0.4939
        self.assertEqual(expected_weight, computed_weight)

class BroadcastNetworkExtractionTest(unittest.TestCase):
    # =========================================================================
    # ================= extract_network_from_broadcast ========================
    # =========================================================================
    @classmethod
    def setUpClass(cls):
        cls.data = pd.DataFrame({
        'event_content': [
            'hi there',
            'hello my friend',
            'what\'s up buddy?',
            'nothing',
            'hey prick!!!'],
        'timestamp': ['2019-05-21 10:17:36',
                      '2019-05-21 10:17:37',
                      '2019-05-21 10:18:05',
                      '2019-05-21 10:18:17',
                      '2019-05-21 10:25:36'],
        'sender_subject_id': [123, 122, 123, 122, 124]})
        cls.extractor = bne.NetworkExtraction()
        cls.real_log_data = pd.DataFrame({
            "id": [3677, 3679, 3680, 3682],
            "event_type": [
                'COMMUNICATION_MESSAGE', 'COMMUNICATION_MESSAGE',
                'COMMUNICATION_MESSAGE', 'COMMUNICATION_MESSAGE'],
            "event_content": ["agree on radio waves? ", "yes", "yess", "yeah"],
            "timestamp": [
                '2019-05-13 10:40:15', '2019-05-13 10:40:18	',
                '2019-05-13 10:40:18', '2019-05-13 10:40:19'],
            "completed_task_id": [52, 52, 52, 52],
            "sender_subject_id": [29, 30, 32, 31],
            "receiver_subject_id": ["\\\\N", "\\\\N", "\\\\N", "\\\\N"],
            "session_id": [6, 6, 6, 6],
            "sender": ["pogs5.1", "pogs5.2", "pogs5.4", "pogs5.3"],
            "receiver": ["\\\\N", "\\\\N", "\\\\N", "\\\\N"],
            "extra_data": ["\\\\N", "\\\\N", "\\\\N", "\\\\N"],
            "team_id": [11, 11, 11, 11]})
        cls.back_and_forth_data = pd.DataFrame({
        'event_content': [
            'Do we have an agreement on the answer?',  # 1
            'Yes',                                     # 2
            'Totally.',                                # 3
            'Can I send it?',                          # 3
            'Yeah go for it.',                         # 1
            'I agree too.'],                           # 2
        'timestamp': ['2019-05-21 10:17:00',   # 1
                      '2019-05-21 10:17:03',   # 2
                      '2019-05-21 10:17:04',   # 3
                      '2019-05-21 10:17:07',   # 3
                      '2019-05-21 10:17:11',   # 1
                      '2019-05-21 10:17:12'],  # 2
        'sender_subject_id': [1, 2, 3, 3, 1, 2]})

    def test_extract_network_from_broadcast_returns_empty_graph_when_empty_df(
        self):
        empty_data = pd.DataFrame({
            'event_content': [],
            'timestamp': [],
            'sender_subject_id': []})
        computed_graph = self.extractor.extract_network_from_broadcast(
            communication_data=empty_data)
        expected_graph = nx.DiGraph()
        utils.assert_graph_equals(computed_graph, expected_graph)

    def test_extract_network_from_broadcast_raises_when_wrong_time_window(
        self):
        with self.assertRaises(ValueError):
            self.extractor.extract_network_from_broadcast(
                communication_data=self.data,
                time_window=[9, 2])

    def test_extract_network_from_broadcast_raises_when_missing_column(self):
        with self.assertRaises(ValueError):
            self.extractor.extract_network_from_broadcast(
                communication_data=self.data,
                column_names=bne.ColumnNameOptions(
                    text_column_name='not_existing_column_name'))

    ########################### On the data log ###############################
    def test_extract_network_from_broadcast_unweighted(self):
        computed_graph = self.extractor.extract_network_from_broadcast(
            communication_data=self.data,
            time_window=[1, 20],
            weight_type=bne.WeightType.NONE,
            aggregation_type=bne.AggregationType.BINARY)
        expected_graph = nx.DiGraph()
        expected_graph.add_nodes_from([122, 123])
        expected_graph.add_edge(122, 123, weight=1)
        utils.assert_graph_equals(computed_graph, expected_graph)

    def test_extract_network_from_broadcast_count_weight(self):
        computed_graph = self.extractor.extract_network_from_broadcast(
            communication_data=self.data,
            time_window=[1, 20],
            weight_type=bne.WeightType.NONE,
            aggregation_type=bne.AggregationType.SUM)
        expected_graph = nx.DiGraph()
        expected_graph.add_nodes_from([122, 123])
        expected_graph.add_edge(122, 123, weight=2.0)
        utils.assert_graph_equals(computed_graph, expected_graph)

    def test_extract_network_count_same_as_sum_reply_duration_weight(self):
        # It tests whether network with no weight and sum is same as with the
        #   one computed via exponential sum with gamma coefficient equals 0.
        computed_graph1 = self.extractor.extract_network_from_broadcast(
            communication_data=self.data,
            time_window=[1, 20],
            weight_type=bne.WeightType.REPLY_DURATION,
            aggregation_type=bne.AggregationType.SUM,
            gamma=0)
        computed_graph2 = self.extractor.extract_network_from_broadcast(
            communication_data=self.data,
            time_window=[1, 20],
            weight_type=bne.WeightType.NONE,
            aggregation_type=bne.AggregationType.SUM)
        utils.assert_graph_equals(computed_graph1, computed_graph2)

    def test_extract_network_from_broadcast_average_reply_duration_weight(
        self):
        computed_graph = self.extractor.extract_network_from_broadcast(
            communication_data=self.data,
            time_window=[1, 20],
            weight_type=bne.WeightType.REPLY_DURATION,
            aggregation_type=bne.AggregationType.AVERAGE,
            gamma=0.15)
        expected_graph = nx.DiGraph()
        expected_graph.add_nodes_from([122, 123])
        expected_graph.add_edge(122, 123, weight=0.5130034323233221)
        utils.assert_graph_equals(computed_graph, expected_graph)

    def test_extract_network_from_broadcast_average_sentiment_weight(self):
        computed_graph = self.extractor.extract_network_from_broadcast(
            communication_data=self.data,
            time_window=[1, 20],
            weight_type=bne.WeightType.SENTIMENT,
            aggregation_type=bne.AggregationType.AVERAGE)
        expected_graph = nx.DiGraph()
        expected_graph.add_nodes_from([122, 123])
        expected_graph.add_edge(122, 123, weight=0.24695)
        utils.assert_graph_equals(computed_graph, expected_graph)

    def test_extract_network_from_broadcast_average_emotion_valence_weight(
        self):
        computed_graph = self.extractor.extract_network_from_broadcast(
            communication_data=self.data,
            time_window=[1, 20],
            weight_type=bne.WeightType.EMOTION_VALENCE,
            aggregation_type=bne.AggregationType.AVERAGE)
        expected_graph = nx.DiGraph()
        expected_graph.add_nodes_from([122, 123])
        expected_graph.add_edge(122, 123, weight=3.87)
        utils.assert_graph_equals(computed_graph, expected_graph)
    
    def test_extract_network_from_broadcast_average_emotion_arousal_weight(
        self):
        computed_graph = self.extractor.extract_network_from_broadcast(
            communication_data=self.data,
            time_window=[1, 20],
            weight_type=bne.WeightType.EMOTION_AROUSAL,
            aggregation_type=bne.AggregationType.AVERAGE)
        expected_graph = nx.DiGraph()
        expected_graph.add_nodes_from([122, 123])
        expected_graph.add_edge(122, 123, weight=2.87)
        utils.assert_graph_equals(computed_graph, expected_graph)

    def test_extract_network_from_broadcast_average_emotion_dominance_weight(
        self):
        computed_graph = self.extractor.extract_network_from_broadcast(
            communication_data=self.data,
            time_window=[1, 20],
            weight_type=bne.WeightType.EMOTION_DOMINANCE,
            aggregation_type=bne.AggregationType.AVERAGE)
        expected_graph = nx.DiGraph()
        expected_graph.add_nodes_from([122, 123])
        expected_graph.add_edge(122, 123, weight=3.37)
        utils.assert_graph_equals(computed_graph, expected_graph)

    ####################### On the real log data ##############################
    def test_extract_network_from_broadcast_unweighted_real_log_bad_window(
        self):
        computed_graph = self.extractor.extract_network_from_broadcast(
            communication_data=self.real_log_data,
            time_window=[1, 10],
            weight_type=bne.WeightType.NONE,
            aggregation_type=bne.AggregationType.BINARY)
        expected_graph = nx.DiGraph()
        expected_graph.add_nodes_from([30, 29, 32, 31])
        expected_graph.add_edge(30, 29, weight=1) # 30 answers to person 29
        expected_graph.add_edge(32, 29, weight=1) # 32 answers to person 29
        expected_graph.add_edge(31, 29, weight=1) # 31 answers to person 29
        expected_graph.add_edge(31, 30, weight=1) # An unfortunate answer!
        expected_graph.add_edge(31, 32, weight=1) # An unfortunate answer!
        utils.assert_graph_equals(computed_graph, expected_graph)

    def test_extract_network_from_broadcast_unweighted_real_log(self):
        computed_graph = self.extractor.extract_network_from_broadcast(
            communication_data=self.real_log_data,
            time_window=[2, 10],
            weight_type=bne.WeightType.NONE,
            aggregation_type=bne.AggregationType.BINARY)
        expected_graph = nx.DiGraph()
        expected_graph.add_nodes_from([30, 29, 32, 31])
        expected_graph.add_edge(30, 29, weight=1) # 30 answers to person 29
        expected_graph.add_edge(32, 29, weight=1) # 32 answers to person 29
        expected_graph.add_edge(31, 29, weight=1) # 31 answers to person 29
        utils.assert_graph_equals(computed_graph, expected_graph)

    def test_extract_network_from_broadcast_average_sentiment_weight_real_log(
        self):
        computed_graph = self.extractor.extract_network_from_broadcast(
            communication_data=self.real_log_data,
            time_window=[2, 10],
            weight_type=bne.WeightType.SENTIMENT,
            aggregation_type=bne.AggregationType.AVERAGE)
        expected_graph = nx.DiGraph()
        expected_graph.add_nodes_from([30, 29, 32, 31])
        expected_graph.add_edge(30, 29, weight=0.4019)   # Sentiment of yes.
        expected_graph.add_edge(
            32, 29, weight=0.0)  # If English is not fixed, sentiment is 0!
        expected_graph.add_edge(31, 29, weight=0.296)    # Sentiment of yeah.
        utils.assert_graph_equals(computed_graph, expected_graph)

    def test_extract_network_from_broadcast_average_reply_dur_weight_real_log(
        self):
        computed_graph = self.extractor.extract_network_from_broadcast(
            communication_data=self.real_log_data,
            time_window=[1, 10],
            weight_type=bne.WeightType.REPLY_DURATION,
            aggregation_type=bne.AggregationType.AVERAGE,
            gamma=0.15)
        expected_graph = nx.DiGraph()
        expected_graph.add_nodes_from([30, 29, 32, 31])
        expected_graph.add_edge(
            30, 29, weight=np.exp(-0.15*3))  # 30 answers to 29.
        expected_graph.add_edge(
            32, 29, weight=np.exp(-0.15*3))  # 32 answers to 29
        expected_graph.add_edge(
            31, 29, weight=np.exp(-0.15*4))  # 31 answers to 29
        expected_graph.add_edge(31, 30, weight=np.exp(-0.15*1))
        expected_graph.add_edge(31, 32, weight=np.exp(-0.15*1))
        utils.assert_graph_equals(computed_graph, expected_graph)

    ############# On the back and forth log data ##############################
    def test_extract_network_from_broadcast_unweighted_backandforth_log(self):
        computed_graph = self.extractor.extract_network_from_broadcast(
            communication_data=self.back_and_forth_data,
            time_window=[2, 10],
            weight_type=bne.WeightType.NONE,
            aggregation_type=bne.AggregationType.BINARY)
        expected_graph = nx.DiGraph()
        expected_graph.add_nodes_from([1, 2, 3])
        expected_graph.add_edge(2, 1, weight=1)
        expected_graph.add_edge(3, 1, weight=1)
        expected_graph.add_edge(3, 2, weight=1)
        expected_graph.add_edge(1, 2, weight=1)
        expected_graph.add_edge(1, 3, weight=1)
        expected_graph.add_edge(2, 3, weight=1)
        utils.assert_graph_equals(computed_graph, expected_graph)

    def test_extract_network_from_broadcast_count_weight_backandforth_log(self):
        computed_graph = self.extractor.extract_network_from_broadcast(
            communication_data=self.back_and_forth_data,
            time_window=[2, 10],
            weight_type=bne.WeightType.NONE,
            aggregation_type=bne.AggregationType.SUM)
        expected_graph = nx.DiGraph()
        expected_graph.add_nodes_from([1, 2, 3])
        expected_graph.add_edge(2, 1, weight=1)
        expected_graph.add_edge(3, 1, weight=2)
        expected_graph.add_edge(3, 2, weight=1)
        expected_graph.add_edge(1, 2, weight=1)
        expected_graph.add_edge(1, 3, weight=2)
        expected_graph.add_edge(2, 3, weight=2)
        utils.assert_graph_equals(computed_graph, expected_graph)

    def test_extract_network_reply_duration_weight_backandforth_log(self):
        computed_graph = self.extractor.extract_network_from_broadcast(
            communication_data=self.back_and_forth_data,
            time_window=[2, 10],
            weight_type=bne.WeightType.REPLY_DURATION,
            aggregation_type=bne.AggregationType.AVERAGE,
            gamma=0.1)
        expected_graph = nx.DiGraph()
        expected_graph.add_nodes_from([1, 2, 3])
        expected_graph.add_edge(2, 1, weight=np.exp(-0.1*3))
        expected_graph.add_edge(
            3, 1, weight=(np.exp(-0.1*4) + np.exp(-0.1*7))/2)
        expected_graph.add_edge(3, 2, weight=np.exp(-0.1*4))
        expected_graph.add_edge(1, 2, weight=np.exp(-0.1*8))
        expected_graph.add_edge(
            1, 3, weight=(np.exp(-0.1*7) + np.exp(-0.1*4))/2)
        expected_graph.add_edge(
            2, 3, weight=(np.exp(-0.1*8) + np.exp(-0.1*5))/2)
        utils.assert_graph_equals(computed_graph, expected_graph)


    def test_extract_network_from_broadcast_sentiment_weight_backandforth_log(
        self):
        computed_graph = self.extractor.extract_network_from_broadcast(
            communication_data=self.back_and_forth_data,
            time_window=[2, 10],
            weight_type=bne.WeightType.SENTIMENT,
            aggregation_type=bne.AggregationType.AVERAGE)
        expected_graph = nx.DiGraph()
        expected_graph.add_nodes_from([1, 2, 3])
        expected_graph.add_edge(2, 1, weight=0.4019)
        expected_graph.add_edge(3, 1, weight=0)
        expected_graph.add_edge(3, 2, weight=0)
        expected_graph.add_edge(1, 2, weight=0.296)
        expected_graph.add_edge(1, 3, weight=0.592/2)
        expected_graph.add_edge(2, 3, weight=0.7224/2)
        utils.assert_graph_equals(computed_graph, expected_graph)
