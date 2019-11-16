# Omid55
# Test module for broadcast_network_extraction.

from __future__ import division, print_function, absolute_import, unicode_literals

import pandas as pd
import networkx as nx
import unittest
from parameterized import parameterized
import utils

import broadcast_network_extraction as bne


class BroadcastNetworkExtractionTest(unittest.TestCase):
    # =========================================================================
    # =========================== extract_network =============================
    # =========================================================================
    def test_extract_network(self):
        data = pd.DataFrame({
            'content': ['hi', 'hello', 'what\'s up?', 'nothing', 'hey!!!'],
            'timestamp': ['2019-05-21 10:17:36',
                          '2019-05-21 10:17:37',
                          '2019-05-21 10:18:05',
                          '2019-05-21 10:18:17',
                          '2019-05-21 10:25:36'],
            'sender_subject_id': [123, 122, 123, 122, 124]
        })
        computed_graph = bne.extract_network(
            communication_data=data,
            text_column_name='content',
            time_column_name='timestamp',
            sender_column_name='sender_subject_id',
            time_window=[1, 20],
            weight_type=bne.WeightType.NUMBER_OF_MESSAGES
        )
        expected_graph = nx.DiGraph()
        expected_graph.add_nodes_from([122, 123])
        expected_graph.add_edge(122, 123, weight=2.0)
        utils.assert_graph_equals(computed_graph, expected_graph)
