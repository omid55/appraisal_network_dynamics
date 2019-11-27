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

import attr
import enum
import numpy as np
import pandas as pd
import networkx as nx
from typing import List
from typing import Text
from typing import Tuple
from datetime import timedelta

import text_processor
import utils


class Error(Exception):
    """Base Error class for this module."""


@attr.s
class ColumnNameOptions(object):
    """Column name options for messages dataframe."""

    # Name of the text column.
    text_column_name = attr.ib(default='event_content')

    # Name of the time column.
    time_column_name = attr.ib(default='timestamp')

    # Name of the sender column.
    sender_column_name = attr.ib(default='sender_subject_id')


class WeightType(enum.Enum):
    """Type of weights to be computed for the network edges.
    
    NONE: means no edge weight is needed (existence of edge or not matter).

    REPLY_DURATION: weight is computed based on the duration of reply. The
      faster the reply has been received higher the weight is. However, there is
      a window applied to withdraw very quick or very late replies. Very quick
      replies might be just coincidental.

    SENTIMENT: weight of each edge is computed using a sentiment analysis model.

    EMOTION_VALENCE: weight of each edge is based on valence (ranging from pleasant to unpleasant).
    
    EMOTION_AROUSAL: weight of each edge is based on arousal (ranging from calm to excited).

    EMOTION_DOMINANCE: weight of each edge is based on dominance (level of the control).
    """
    NONE = 1
    REPLY_DURATION = 2
    SENTIMENT = 3
    EMOTION_VALENCE = 4
    EMOTION_AROUSAL = 5
    EMOTION_DOMINANCE = 6


class AggregationType(enum.Enum):
    BINARY = 1
    SUM = 2
    AVERAGE = 3
    LAST = 4
    MAX = 5
    MIN = 6

    @staticmethod
    def _my_exists(input_list: List) -> int:
        if len(input_list) > 0:
            return 1
        return 0

    @staticmethod
    def _my_last(input_list: List) -> float:
        if len(input_list) == 0:
            raise ValueError('List input for aggregator last was empty.')
        return input_list[-1]

    def get_function(self):
        """Gets the python understandable function to be used on list of values.

        Args:
            None.

        Returns:
            A python builtin or self written function.

        Raises:
            None.
        """
        enum2function = {
            'AggregationType.BINARY': AggregationType._my_exists,
            'AggregationType.SUM': sum,
            'AggregationType.AVERAGE': np.mean,
            'AggregationType.LAST': AggregationType._my_last,
            'AggregationType.MAX': max,
            'AggregationType.MIN': min}
        return enum2function[str(self)]


class NetworkExtraction(object):
    def __init__(self):
        self.sentiment_analyzer = text_processor.SentimentAnalyzer()
        self.emotion_analyzer = text_processor.EmotionDetector()

    def _compute_weight(self,
                        time1: pd.Timestamp,
                        time2: pd.Timestamp,
                        weight_type: WeightType,
                        content: Text = '',
                        gamma: float = 0) -> float:
        """Computes the weight of the edge from a message based on time/content.

        Args:
            time1: The time of message by sender.

            time2: The time of message by responder which should be after time1.

            weight_type: The type of edge weight we want.

            content: The text content of the message.

            gamma: The coefficient for reply duration computation. 

        Returns:
            Edge weight value.

        Raises:
            ValueError: If time2 was after time1.
        """
        if weight_type == WeightType.NONE:
            return 1.0
        elif weight_type == WeightType.REPLY_DURATION:
            duration = (time2 - time1).total_seconds()
            if duration < 0:
                raise ValueError(
                    'Second time cannot be smaller than the first one.')
            return np.exp(-gamma * duration)
        elif weight_type == WeightType.SENTIMENT:
            return self.sentiment_analyzer.compute_sentiment(content)
        elif weight_type == WeightType.EMOTION_VALENCE:
            return self.emotion_analyzer.compute_mean_emotion(content)[1]
        elif weight_type == WeightType.EMOTION_AROUSAL:
            return self.emotion_analyzer.compute_mean_emotion(content)[2]
        elif weight_type == WeightType.EMOTION_DOMINANCE:
            return self.emotion_analyzer.compute_mean_emotion(content)[3]
        else:
            raise ValueError('Wrong weight type was sent.')

    def _apply_function_on_all_edges(self,
                                     dgraph: nx.DiGraph,
                                     aggregation_type: AggregationType
                                     ) -> nx.DiGraph:
        """Applies a function on graph edges with list of values as their weights.

        Args:
            dgraph: Directed graph with a list of float weight vals on its edges.

            aggregation_type: Type of aggregation to be applied on values.

        Returns:
            A new directed graph with weights to be float values.

        Raises:
            ValueError: If the list was empty when aggregator is last.
        """
        new_dgraph = dgraph.copy()
        for edge in dgraph.edges():
            weight_list = dgraph[edge[0]][edge[1]]['weight']
            func = aggregation_type.get_function()
            new_dgraph[edge[0]][edge[1]]['weight'] = func(weight_list)
        return new_dgraph

    def extract_network_from_broadcast(
        self,
        communication_data: pd.DataFrame,
        time_window: Tuple = [1, 10],
        weight_type: WeightType = WeightType.NONE,
        aggregation_type: AggregationType = AggregationType.SUM,
        column_names: ColumnNameOptions = ColumnNameOptions(),
        gamma: float = 0.0) -> nx.DiGraph:
        """Extracts the network structure of members in a broadcast setting's log.

        Args:
            communication_data: Dataframe of a broadcast log.

            time_window: The time window in seconds to be considered a reply.

            weight_type: Type of weight to be considered for the network. 

            aggregation_type: Type of aggregation on edges' weights.

            column_names: The column names options needed in communication_data.

            gamma: The coefficient for exponential weight based on duration.

        Returns:
            A directed graph structure among members in a broadcast log data.

        Raises:
            ValueError: If the column names do not exist in the communication data.
        """
        # Opens the settings for the convenience.
        # ------------------------------------------------
        text_column_name = column_names.text_column_name
        time_column_name = column_names.time_column_name
        sender_column_name = column_names.sender_column_name
        # ------------------------------------------------
        utils.check_required_columns(
            communication_data, [
                text_column_name, time_column_name, sender_column_name])
        dgraph = nx.DiGraph()
        if isinstance(communication_data.iloc[0][time_column_name], str):
            communication_data[time_column_name] = (
                pd.to_datetime(communication_data[time_column_name]))
        for j in communication_data.index:
            pivot_time = communication_data.at[j, time_column_name]
            pivot = communication_data.at[j, sender_column_name]
            response_df = communication_data[
                (communication_data[time_column_name] - pivot_time).between(
                    timedelta(seconds=time_window[0]),
                    timedelta(seconds=time_window[1]))]
            for i in response_df.index:
                responder = response_df.at[i, sender_column_name]
                response_time = response_df.at[i, time_column_name]
                response_text = response_df.at[i, text_column_name]
                if not dgraph.has_edge(responder, pivot):
                    dgraph.add_edge(
                        responder, pivot, weight=[
                            self._compute_weight(
                                time1=pivot_time,
                                time2=response_time,
                                weight_type=weight_type,
                                content=response_text,
                                gamma=gamma)])
                else:
                    weight = dgraph[responder][pivot]['weight']
                    dgraph[responder][pivot]['weight'] = weight + [
                        self._compute_weight(
                            time1=pivot_time,
                            time2=response_time,
                            weight_type=weight_type,
                            content=response_text,
                            gamma=gamma)]
        dgraph = self._apply_function_on_all_edges(
            dgraph=dgraph, aggregation_type=aggregation_type)
        return dgraph
