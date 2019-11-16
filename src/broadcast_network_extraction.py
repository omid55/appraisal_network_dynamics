# Omid55
# Start date:     3 Oct 2019
# Modified date:  25 Oct 2019
# Author:   Omid Askarisichani
# Email:    omid55@cs.ucsb.edu
#
# Module for extracting network object for a broadcast communication log.
# Example:
#

from __future__ import division, print_function, absolute_import, unicode_literals

import enum
import numpy as np
import pandas as pd
import networkx as nx
from typing import Text
from typing import Tuple
from datetime import timedelta
import utils


class Error(Exception):
    """Base Error class for this module."""


class WeightType(enum.Enum):
    NONE = 1
    NUMBER_OF_MESSAGES = 2
    REPLY_DURATION = 3
    SENTIMENT = 4
    # EMOTION = 5

class AggregationType(enum.Enum):
    SUM = 1
    AVERAGE = 2
    LAST = 3
    MAX = 4
    MIN = 5

def _compute_weight(time1, time2, text: Text) -> float:
    return 1.0


def _apply_function_on_all_edges(
    dgraph: nx.DiGraph, aggregation_type: AggregationType) -> nx.DiGraph:
    new_dgraph = dgraph.copy()
    for edge in dgraph.edges():
        value = dgraph[edge[0]][edge[1]]['weight']
        if aggregation_type == AggregationType.SUM:
            func = sum
        new_dgraph[edge[0]][edge[1]]['weight'] = func(value)
    return new_dgraph


def extract_network(communication_data: pd.DataFrame,
                    text_column_name: Text,
                    time_column_name: Text,
                    sender_column_name: Text,
                    time_window: Tuple = [1, 10],
                    weight_type: WeightType = WeightType.NONE,
                    aggregation_type: AggregationType = AggregationType.SUM
                    ) -> nx.DiGraph:
    """Extracts the network structure of members in broadcast log.

    Args:

    Returns:

    Raises:
    """
    utils.check_required_columns(
        communication_data, [
            text_column_name, time_column_name, sender_column_name])
    dgraph = nx.DiGraph()
    if isinstance(communication_data.iloc[0][time_column_name], str):
        communication_data[time_column_name] = (
            pd.to_datetime(communication_data[time_column_name]))
    for _, row in communication_data.iterrows():
        pivot_time = row[time_column_name]
        pivot = row[sender_column_name]
        response_df = communication_data[
            (communication_data[time_column_name] - pivot_time).between(
                timedelta(seconds=time_window[0]),
                timedelta(seconds=time_window[1]))]
        for _, response_row in response_df.iterrows():
            responder = response_row[sender_column_name]
            response_time = response_row[time_column_name]
            response_text = response_row[text_column_name]
            if not dgraph.has_edge(responder, pivot):
                dgraph.add_edge(
                    responder, pivot, weight=[
                        _compute_weight(
                            pivot_time, response_time, response_text)])
            elif weight_type != WeightType.NONE:
                weight = dgraph[responder][pivot]['weight']
                dgraph[responder][pivot]['weight'] = weight + [
                    _compute_weight(
                        pivot_time, response_time, response_text)]
    dgraph = _apply_function_on_all_edges(
        dgraph=dgraph, aggregation_type=aggregation_type)
    return dgraph
