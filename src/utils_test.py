# Omid55
# Test module for network_utils.

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import numpy.testing as np_testing
import unittest
import sys
import os
from parameterized import parameterized
from io import StringIO

import utils


class MyTestClass(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.simple_dataframe = pd.DataFrame(
            {'col1': [1, 2], 'col2': [3, 4], 'col3': [5, 6]})
        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3, 4])
        dg.add_edge(1, 2, weight=1)
        dg.add_edge(1, 3, weight=-1)
        dg.add_edge(2, 3, weight=5)
        dg.add_edge(3, 1, weight=-4)
        dg.add_edge(4, 1, weight=2)
        cls.sample_dgraph = dg

    @classmethod
    def tearDown(cls):
        del cls.simple_dataframe

    # =========================================================================
    # ==================== print_dict_pretty ==================================
    # =========================================================================
    def test_print_dict_pretty(self):
        capturedOutput = StringIO()
        sys.stdout = capturedOutput
        sample_dict = {'key1': 'value1', 'key2': 'value2'}
        utils.print_dict_pretty(sample_dict)
        sys.stdout = sys.__stdout__
        computed = capturedOutput.getvalue()
        expected = 'key1: value1\nkey2: value2\n'
        self.assertEqual(expected, computed)

    # =========================================================================
    # ==================== check_required_columns =============================
    # =========================================================================
    def test_if_check_required_columns_not_raise(self):
        utils.check_required_columns(self.simple_dataframe, ['col1', 'col3'])

    def test_if_check_required_columns_raises_when_missing(self):
        with self.assertRaises(ValueError):
            utils.check_required_columns(
                self.simple_dataframe, ['col1', 'col4'])

    # =========================================================================
    # ==================== graph_equals =======================================
    # =========================================================================
    def test_graph_equals_when_same(self):
        g1 = nx.DiGraph()
        g1.add_nodes_from([1, 2, 3])
        g1.add_edge(1, 2)
        g1.add_edge(2, 3)
        g2 = nx.DiGraph()
        g2.add_nodes_from([1, 2, 3])
        g2.add_edge(2, 3)
        g2.add_edge(1, 2)
        self.assertTrue(utils.graph_equals(g1, g2, weight_column_name=None))

    def test_graph_equals_when_not_same_nodes(self):
        g1 = nx.DiGraph()
        g1.add_nodes_from([1, 2, 3])
        g1.add_edge(1, 2)
        g1.add_edge(2, 3)
        g2 = nx.DiGraph()
        g2.add_nodes_from([1, 2, 3])
        g2.add_edge(1, 2)
        g2.add_edge(2, 3)
        g2.add_edge(1, 3)
        self.assertFalse(utils.graph_equals(g1, g2, weight_column_name=None))

    def test_graph_equals_when_not_same_edges(self):
        g1 = nx.DiGraph()
        g1.add_nodes_from([1, 2, 3, 4])
        g1.add_edge(1, 2)
        g1.add_edge(2, 3)
        g2 = nx.DiGraph()
        g2.add_nodes_from([1, 2, 3])
        g2.add_edge(1, 2)
        g2.add_edge(2, 3)
        self.assertFalse(utils.graph_equals(g1, g2, weight_column_name=None))

    def test_graph_equals_when_different_edge_weights(self):
        g1 = nx.DiGraph()
        g1.add_nodes_from([1, 2, 3])
        g1.add_edge(1, 2, weight='5')
        g1.add_edge(2, 3, weight='-1')
        g1.add_edge(1, 3, weight='1')
        g2 = nx.DiGraph()
        g2.add_nodes_from([1, 2, 3])
        g2.add_edge(1, 2, weight='5')
        g2.add_edge(2, 3, weight='9')
        g2.add_edge(1, 3, weight='1')
        self.assertFalse(
            utils.graph_equals(g1, g2, weight_column_name='weight'))

    # =========================================================================
    # ==================== assert_graph_equals ================================
    # =========================================================================
    def test_assert_graph_equals_when_same(self):
        g1 = nx.DiGraph()
        g1.add_nodes_from([1, 2, 3])
        g1.add_edge(1, 2)
        g1.add_edge(2, 3)
        g2 = nx.DiGraph()
        g2.add_nodes_from([1, 2, 3])
        g2.add_edge(2, 3)
        g2.add_edge(1, 2)
        utils.assert_graph_equals(g1, g2, weight_column_name=None)

    def test_assert_graph_equals_when_not_same_nodes(self):
        g1 = nx.DiGraph()
        g1.add_nodes_from([1, 2, 3, 4])
        g1.add_edge(1, 2)
        g1.add_edge(2, 3)
        g2 = nx.DiGraph()
        g2.add_nodes_from([1, 2, 3])
        g2.add_edge(1, 2)
        g2.add_edge(2, 3)
        with self.assertRaises(AssertionError):
            utils.assert_graph_equals(g1, g2, weight_column_name=None)

    def test_assert_graph_equals_when_not_same_edges(self):
        g1 = nx.DiGraph()
        g1.add_nodes_from([1, 2, 3])
        g1.add_edge(1, 2)
        g1.add_edge(2, 3)
        g2 = nx.DiGraph()
        g2.add_nodes_from([1, 2, 3])
        g2.add_edge(1, 2)
        g2.add_edge(2, 3)
        g2.add_edge(1, 3)
        with self.assertRaises(AssertionError):
            utils.assert_graph_equals(g1, g2, weight_column_name=None)

    def test_assert_graph_equals_when_different_edge_weights(self):
        g1 = nx.DiGraph()
        g1.add_nodes_from([1, 2, 3])
        g1.add_edge(1, 2, weight='5')
        g1.add_edge(2, 3, weight='-1')
        g1.add_edge(1, 3, weight='1')
        g2 = nx.DiGraph()
        g2.add_nodes_from([1, 2, 3])
        g2.add_edge(1, 2, weight='5')
        g2.add_edge(2, 3, weight='9')
        g2.add_edge(1, 3, weight='1')
        with self.assertRaises(AssertionError):
            utils.assert_graph_equals(g1, g2, weight_column_name='weight')

    # =========================================================================
    # ==================== sub_adjacency_matrix ===============================
    # =========================================================================
    def test_sub_adjacency_matrix(self):
        adj_matrix = np.arange(16).reshape(4, 4)
        expected = np.array(
            [[0, 1, 3],
             [4, 5, 7],
             [12, 13, 15]])
        computed = utils.sub_adjacency_matrix(adj_matrix, [0, 1, 3])
        np_testing.assert_array_equal(expected, computed)

    # =========================================================================
    # ==================== swap_nodes_in_matrix ===============================
    # =========================================================================
    def test_swap_nodes_in_matrix(self):
        matrix = np.array(
            [[0, 1, 2],
             [3, 4, 5],
             [6, 7, 8]])
        node1 = 0
        node2 = 2
        expected = np.array(
            [[8, 7, 6],
             [5, 4, 3],
             [2, 1, 0]])
        computed = utils.swap_nodes_in_matrix(matrix, node1, node2)
        np_testing.assert_array_equal(expected, computed)

    # =========================================================================
    # ==================== make_matrix_row_stochastic =========================
    # =========================================================================
    def test_make_matrix_row_stochastic(self):
        matrix = np.array(
            [[0, 1, 2],
             [3, 4, 5],
             [6, 7, 8]])
        expected = np.array(
            [[0, 0.33, 0.67],
             [0.25, 0.33, 0.42],
             [0.29, 0.33, 0.38]])
        computed = utils.make_matrix_row_stochastic(matrix)
        np_testing.assert_array_almost_equal(expected, computed, decimal=2)

    def test_make_matrix_row_stochastic_when_mostly_is_row_stochastic_already(
        self):
        matrix = np.matrix([
            [0.85, 0.05, 0.05, 0.05],
            [0.35, 0.25, 0.35, 0.03],
            [0.02, 0.01, 0.02, 0.02],
            [0.25, 0.25, 0.25, 0.25]])
        expected = np.matrix([
            [0.85, 0.05, 0.05, 0.05],
            [0.36, 0.26, 0.36, 0.03],
            [0.29, 0.14, 0.29, 0.29],
            [0.25, 0.25, 0.25, 0.25]])
        computed = utils.make_matrix_row_stochastic(matrix)
        np_testing.assert_array_almost_equal(expected, computed, decimal=2)

    def test_make_matrix_row_stochastic_when_all_zeros(self):
        matrix = np.zeros((4, 4))
        expected = np.ones((4, 4)) * 0.25
        computed = utils.make_matrix_row_stochastic(matrix)
        np_testing.assert_array_almost_equal(expected, computed, decimal=2)

    def test_make_matrix_row_stochastic_when_already_row_stochastic(self):
        matrix = np.matrix([
            [0.11, 0.26, 0.34, 0.29],
            [0.26, 0.21, 0.25, 0.28],
            [0.05, 0.05, 0.85, 0.05],
            [0.25, 0.25, 0.25, 0.25]])
        expected = matrix
        computed = utils.make_matrix_row_stochastic(matrix)
        np_testing.assert_array_almost_equal(expected, computed, decimal=2)

    # =========================================================================
    # ======================= save_figure =====================================
    # ======================= load_figure =====================================
    # =========================================================================
    def test_save_and_load_figure(self):
        fig_object = plt.figure()
        file_path = 'test_file'
        utils.save_figure(fig_object=fig_object, file_path=file_path)
        loaded_fig_object = utils.load_figure(file_path=file_path)
        os.remove(file_path+'.pkl')
        os.remove(file_path+'.pdf')
        self.assertEqual(fig_object.images, loaded_fig_object.images)
        self.assertEqual(fig_object.axes, loaded_fig_object.axes)

    # =========================================================================
    # ============= save_all_variables_of_current_session =====================
    # ============= load_all_variables_of_saved_session =======================
    # =========================================================================
    def test_save_and_load_all_variables_firstpart(self):
        str_var = 'anything'
        list_var = [1, 2, 5]
        file_path = 'test_file'
        utils.save_all_variables_of_current_session(locals(), file_path)

    def test_save_and_load_all_variables_secondpart(self):
        expected_str_var = 'anything'
        expected_list_var = [1, 2, 5]
        file_path = 'test_file'
        utils.load_all_variables_of_saved_session(globals(), file_path)
        self.assertEqual(expected_str_var, str_var)
        self.assertEqual(expected_list_var, list_var)
        os.remove(file_path)

    # =========================================================================
    # ==================== swap_two_elements_in_matrix ========================
    # =========================================================================
    @parameterized.expand([
        ['InPlace', True],
        ['NotInPlace', False]])
    def test_swap_two_elements_in_matrix(self, name, inplace):
        matrix = np.array(
            [[0, 1, 2],
             [3, 4, 5],
             [6, 7, 8]])
        expected = np.array(
            [[0, 1, 3],
             [2, 4, 5],
             [6, 7, 8]])
        original_matrix = matrix.copy()
        computed = utils.swap_two_elements_in_matrix(
            matrix=matrix, x1=0, y1=2, x2=1, y2=0, inplace=inplace)
        np_testing.assert_array_equal(expected, computed)
        if inplace:
            self.assertEqual(matrix.all(), computed.all())
        else:
            self.assertEqual(matrix.all(), original_matrix.all())

    # =========================================================================
    # ==================== dgraph2adjacency ===================================
    # =========================================================================
    def test_dgraph2adjacency(self):
        dg = self.sample_dgraph
        expected = np.array(
            [[0, 1, -1, 0],
             [0, 0, 5, 0],
             [-4, 0, 0, 0],
             [2, 0, 0, 0]])
        computed = utils.dgraph2adjacency(dg)
        np_testing.assert_array_equal(expected, computed)

    # =========================================================================
    # ==================== adjacency2digraph ==================================
    # =========================================================================
    def test_adjacency2digraph(self):
        dg = self.sample_dgraph
        adj_matrix = utils.dgraph2adjacency(dg)
        # We make another directed graph with the same adjacency matrix. Thus
        #   the graphs should match.
        computed_graph = utils.adjacency2digraph(
            adj_matrix=adj_matrix, similar_this_dgraph=dg)
        self.assertEqual(dg.nodes(), computed_graph.nodes())
        self.assertEqual(dg.edges(), computed_graph.edges())
        # Checking every edge weight.
        for edge in dg.edges():
            self.assertEqual(
                dg.get_edge_data(edge[0], edge[1]),
                computed_graph.get_edge_data(edge[0], edge[1]))

    def test_adjacency2digraph_without_similar_graph(self):
        dg = self.sample_dgraph
        adj_matrix = utils.dgraph2adjacency(dg)
        # We need to map the node labels to start from 0 due to the default.
        dg = nx.relabel_nodes(dg, mapping={i+1: i for i in range(4)})
        computed_graph = utils.adjacency2digraph(adj_matrix=adj_matrix)
        self.assertEqual(dg.nodes(), computed_graph.nodes())
        self.assertEqual(dg.edges(), computed_graph.edges())
        # Checking every edge weight.
        for edge in dg.edges():
            self.assertEqual(
                dg.get_edge_data(edge[0], edge[1]),
                computed_graph.get_edge_data(edge[0], edge[1]))

    # =========================================================================
    # ================ _adjacency2digraph_with_given_mapping ==================
    # =========================================================================
    def test_adjacency2digraph_with_given_mapping(self):
        dg = self.sample_dgraph
        adj_matrix = utils.dgraph2adjacency(dg)
        node_mapping = {0: 1, 1: 2, 2: 3, 3: 4}
        computed_graph = utils._adjacency2digraph_with_given_mapping(
            adj_matrix=adj_matrix, node_mapping=node_mapping)
        self.assertEqual(dg.nodes(), computed_graph.nodes())
        self.assertEqual(dg.edges(), computed_graph.edges())
        # Checking every edge weight.
        for edge in dg.edges():
            self.assertEqual(
                dg.get_edge_data(edge[0], edge[1]),
                computed_graph.get_edge_data(edge[0], edge[1]))

    # =========================================================================
    # ==================== save_it and load_it ================================
    # =========================================================================
    def test_load_it_and_save_it(self):
        a = [1, 2, 6, 10]
        file_path = 'tmp.pk'
        utils.save_it(a, file_path)
        b = utils.load_it(file_path)
        os.remove(file_path)
        self.assertEqual(a, b)

    # =========================================================================
    # ================= draw_from_empirical_distribution ======================
    # =========================================================================
    def test_draw_from_empirical_distribution_raises_when_not_positive_nbins(self):
        with self.assertRaises(ValueError):
            utils.draw_from_empirical_distribution(
                [1, 2, 3], nbins=-2)

    def test_draw_from_empirical_distribution_raises_when_too_many_nbins(self):
        with self.assertRaises(ValueError):
            utils.draw_from_empirical_distribution(
                data_points=[1, 2, 3], nbins=4)

    def test_draw_from_empirical_distribution_raises_when_empty_data_points(self):
        with self.assertRaises(ValueError):
            utils.draw_from_empirical_distribution(
                data_points=[], nbins=2)
    
    def test_draw_from_empirical_distribution(self):
        expected = 2
        computed = utils.draw_from_empirical_distribution(
            data_points=[2, 2, 2, 2, 2],
            nbins=1)
        self.assertEqual(expected, np.round(computed))

    # =========================================================================
    # ==================== shuffle_matrix_in_given_order ======================
    # =========================================================================
    def test_shuffle_matrix_in_given_order_raises_when_wrong_matrix(self):
        with self.assertRaises(ValueError):
            utils.shuffle_matrix_in_given_order(
                matrix=np.matrix([[1, 2, 3], [4, 5, 6]]),
                order=np.array([0, 2, 1]))

    def test_shuffle_matrix_in_given_order_raises_when_wrong_order_length(self):
        with self.assertRaises(ValueError):
            utils.shuffle_matrix_in_given_order(
                matrix=np.matrix([[1, 2], [3, 4]]),
                order=np.array([0, 2, 1]))

    def test_shuffle_matrix_in_given_order(self):
        matrix = np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        order = np.array([0, 2, 1])
        expected = np.matrix([[1, 3, 2], [7, 9, 8], [4, 6, 5]])
        computed = utils.shuffle_matrix_in_given_order(
            matrix=matrix, order=order)
        np_testing.assert_array_equal(expected, computed)


if __name__ == '__main__':
    unittest.main()
