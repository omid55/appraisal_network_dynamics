# Omid55
# Test module for broadcast_network_extraction.

from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
import pandas as pd
import networkx as nx
import unittest
from parameterized import parameterized
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

class BroadcastNetworkExtractionTest(unittest.TestCase):
    # =========================================================================
    # ================= extract_network_from_broadcast ========================
    # =========================================================================
    @classmethod
    def setUpClass(cls):
        cls.data = pd.DataFrame({
        'event_content': ['hi', 'hello', 'what\'s up?', 'nothing', 'hey!!!'],
        'timestamp': ['2019-05-21 10:17:36',
                      '2019-05-21 10:17:37',
                      '2019-05-21 10:18:05',
                      '2019-05-21 10:18:17',
                      '2019-05-21 10:25:36'],
        'sender_subject_id': [123, 122, 123, 122, 124]})
        cls.extractor = bne.NetworkExtraction()

    def test_extract_network_from_broadcast_unweighted(self):
        computed_graph = self.extractor.extract_network_from_broadcast(
            communication_data=self.data,
            time_window=[1, 20],
            weight_type=bne.WeightType.NONE,
            aggregation_type=bne.AggregationType.BINARY
        )
        expected_graph = nx.DiGraph()
        expected_graph.add_nodes_from([122, 123])
        expected_graph.add_edge(122, 123, weight=1)
        utils.assert_graph_equals(computed_graph, expected_graph)

    def test_extract_network_from_broadcast_count_weights(self):
        computed_graph = self.extractor.extract_network_from_broadcast(
            communication_data=self.data,
            time_window=[1, 20],
            weight_type=bne.WeightType.NONE,
            aggregation_type=bne.AggregationType.SUM
        )
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
            gamma=0,
        )
        computed_graph2 = self.extractor.extract_network_from_broadcast(
            communication_data=self.data,
            time_window=[1, 20],
            weight_type=bne.WeightType.NONE,
            aggregation_type=bne.AggregationType.SUM,
        )
        utils.assert_graph_equals(computed_graph1, computed_graph2)

    def test_extract_network_from_broadcast_average_reply_duration_weight(self):
        computed_graph = self.extractor.extract_network_from_broadcast(
            communication_data=self.data,
            time_window=[1, 20],
            weight_type=bne.WeightType.REPLY_DURATION,
            aggregation_type=bne.AggregationType.AVERAGE,
            gamma=0.15,
        )
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
        expected_graph.add_edge(122, 123, weight=0.0)
        utils.assert_graph_equals(computed_graph, expected_graph)
