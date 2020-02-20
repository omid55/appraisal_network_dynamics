from __future__ import division, print_function, absolute_import, unicode_literals

import pickle as pk
from mytools import Timer
import utils
import broadcast_network_extraction
import pogs_jeopardy_log_lib
import text_processor
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import imp
import networkx as nx
from collections import defaultdict
import sys
sys.path.insert(0, '../src/')


class DatasetBuilder(object):

    def __init__(self, team_log_processors):
        # self.directory = directory
        # self.event_log_directory = event_log_directory
        self.data = {}
        for key in team_log_processors.keys():
            self.data[key] = team_log_processors[key]
        self.time_window = [2, 10]
        self.apply_content_fixer = True
        self.fix_spelling = False
        self.start_index = 0
        self.skip_matrices_not_completely_from_members = False
        self.contents_embeddings = {}
        self.contents_embeddings[7] = {}
        self.contents_embeddings[10] = {}
        # self.contents_embeddings = utils.load_it(
        #     self.directory + 'content_embeddings_with_bert_base.pk')

        self.net_extractor = broadcast_network_extraction.NetworkExtraction()
        self.content_fixer = text_processor.FormalEnglishTranslator(
            './bagofwords/slang.txt')

    def create_dataset(self):
        # self.load_team_logs()
        # self.fix_language_of_messages()
        self.combine_logs()
        self.extract_networks()
        self.extract_contents_of_messages()
        self.extract_individual_performance()
        self.generate_dataset()

    def reload(self):
        imp.reload(pogs_jeopardy_log_lib)
        imp.reload(text_processor)
        imp.reload(utils)
        imp.reload(broadcast_network_extraction)

    def load_data(self):
        self.data = utils.load_it(self.directory+'Teams_logs.pk')

        # Added for now to restrict the data to team 7 and team 10
        small_data = {}
        small_data[7] = self.data[7]
        small_data[10] = self.data[10]
        self.data = small_data

        self.contents = utils.load_it(self.directory+'Teams_contents.pk')
        self.networks = utils.load_it(self.directory+'Teams_networks.pk')
        self.supervised_data = utils.load_it(
            self.directory+'supervised_data.pk')
        self.contents_embeddings = utils.load_it(
            self.directory + 'content_embeddings_with_bert_base.pk')
        self.individual_performance_rates = utils.load_it(
            self.directory + 'Teams_individual_performance.pk')

    def load_team_logs(self):
        with Timer():
            teams = pd.read_csv(
                self.directory+"team.csv",
                sep=',',
                quotechar="|",
                names=["id", "sessionId", "roundId", "taskId"])
            self.data = {}
            for team_id in teams.id:
                print("Processing team", team_id, '...')
                try:
                    self.data[team_id] = pogs_jeopardy_log_lib.TeamLogProcessor(
                        team_id=team_id, logs_directory_path=self.event_log_directory)
                except pogs_jeopardy_log_lib.EventLogsNotLoadedError as e:
                    print(
                        'Team {} is not found in the logs. There is nothing we can do.'.format(team_id))
                    continue
                except Exception as e2:
                    print(e2)
                    print('Team {} had some problems. Check.'.format(team_id))
                    continue

    def fix_language_of_messages(self):
        with Timer():
            if self.apply_content_fixer:
                for team_id, team_log in self.data.items():
                    for i in range(len(team_log.messages)):
                        team_log.messages[i] = self.content_fixer.translate_messages(
                            messages=team_log.messages[i],
                            message_column_name='event_content',
                            fix_spelling=self.fix_spelling)

    def combine_logs(self):
        self.combined_logs = {}
        for team_id, team_log in self.data.items():
            print("Processing team", team_id, '...')
            this_team_number_of_networks = min(
                len(team_log.messages) // 5,
                len(team_log.member_influences))
            all_messages_before_appraisal_reports = []
            for i in range(this_team_number_of_networks):
                all_messages_before_appraisal_reports.append(
                    pd.concat(
                        [team_log.messages[i] for i in np.arange(i * 5, i * 5 + 5)]))
            if len(all_messages_before_appraisal_reports) > 0:
                self.combined_logs[team_id] = all_messages_before_appraisal_reports
            else:
                print('Team', team_id, 'does not have enough logs.')

    def extract_networks(self):
        with Timer():
            self.networks = {}
            for team_id, all_messages_before_appraisal_reports in self.combined_logs.items():
                # if (team_id != 7):
                #     continue
                print("Processing team", team_id, '...')
                this_team_nets = []
                for all_messages_before_appraisal_report in all_messages_before_appraisal_reports:
                    reply_duration_net = self.net_extractor.extract_network_from_broadcast(
                        communication_data=all_messages_before_appraisal_report,
                        time_window=self.time_window,
                        weight_type=broadcast_network_extraction.WeightType.REPLY_DURATION,
                        aggregation_type=broadcast_network_extraction.AggregationType.SUM,
                        gamma=0.15,
                        node_list=self.data[team_id].members)
                    sentiment_net = self.net_extractor.extract_network_from_broadcast(
                        communication_data=all_messages_before_appraisal_report,
                        time_window=self.time_window,
                        weight_type=broadcast_network_extraction.WeightType.SENTIMENT,
                        aggregation_type=broadcast_network_extraction.AggregationType.SUM,
                        node_list=self.data[team_id].members)
                    emotion_arousal_net = self.net_extractor.extract_network_from_broadcast(
                        communication_data=all_messages_before_appraisal_report,
                        time_window=self.time_window,
                        weight_type=broadcast_network_extraction.WeightType.EMOTION_AROUSAL,
                        aggregation_type=broadcast_network_extraction.AggregationType.SUM,
                        node_list=self.data[team_id].members)
                    emotion_dominance_net = self.net_extractor.extract_network_from_broadcast(
                        communication_data=all_messages_before_appraisal_report,
                        time_window=self.time_window,
                        weight_type=broadcast_network_extraction.WeightType.EMOTION_DOMINANCE,
                        aggregation_type=broadcast_network_extraction.AggregationType.SUM,
                        node_list=self.data[team_id].members)
                    emotion_valence_net = self.net_extractor.extract_network_from_broadcast(
                        communication_data=all_messages_before_appraisal_report,
                        time_window=self.time_window,
                        weight_type=broadcast_network_extraction.WeightType.EMOTION_VALENCE,
                        aggregation_type=broadcast_network_extraction.AggregationType.SUM,
                        node_list=self.data[team_id].members)

                    if len(reply_duration_net.nodes()) > 0:
                        this_team_nets.append({
                            'sentiment': sentiment_net,
                            'reply_duration': reply_duration_net,
                            'emotion_arousal': emotion_arousal_net,
                            'emotion_dominance': emotion_dominance_net,
                            'emotion_valence': emotion_valence_net})
                if len(this_team_nets) > 0:
                    self.networks[team_id] = this_team_nets
                else:
                    print('Team', team_id, 'did not have enough networks.')

    def extract_contents_of_messages(self):
        self.contents = {}
        for team_id, all_messages_before_appraisal_reports in self.combined_logs.items():
            print("Processing team", team_id, '...')
            member_concat_messages = []
            for all_messages_before_appraisal_report in all_messages_before_appraisal_reports:
                this_time_member_concat_messages = []
                for member in sorted(self.data[team_id].members):
                    sentences = '[CLS] ' + ' [SEP] '.join(
                        all_messages_before_appraisal_report[
                            all_messages_before_appraisal_report.sender_subject_id == member].event_content) + ' [SEP]'
                    this_time_member_concat_messages.append(sentences)
                member_concat_messages.append(this_time_member_concat_messages)
            self.contents[team_id] = member_concat_messages

    def extract_individual_performance(self):
        self.individual_performance_rates = defaultdict(list)

        hardness_weights = {
            pogs_jeopardy_log_lib.Level.EASY: 1,
            pogs_jeopardy_log_lib.Level.MEDIUM: 2,
            pogs_jeopardy_log_lib.Level.HARD: 3}
        questions = self.data[7].game_info.questions
        for team_id, team_log in self.data.items():
            this_team_members_performance = defaultdict(
                lambda: {'#correct': 0,
                         '#questions': 0,
                         '#hardness_weighted_correct': 0,
                         '#hardness_weighted_questions': 0})

            for index, qid in enumerate(team_log.question_order):
                question_hardness_weight = hardness_weights[questions[qid].level]
                correct_answer = questions[qid].answer
                for member, member_answer in team_log.individual_answers_chosen[qid].items():
                    this_team_members_performance[member]['#questions'] += 1
                    this_team_members_performance[member]['#hardness_weighted_questions'] += question_hardness_weight
                    if member_answer == correct_answer:
                        this_team_members_performance[member]['#correct'] += 1
                        this_team_members_performance[member]['#hardness_weighted_correct'] += question_hardness_weight
                if (index + 1) % 5 == 0:
                    so_far_individual_performance = {}
                    for member in team_log.members:
                        correct_rate = this_team_members_performance[member]['#correct'] / \
                            this_team_members_performance[member]['#questions']
                        hardness_weighted_correct_rate = this_team_members_performance[member][
                            '#hardness_weighted_correct'] / this_team_members_performance[member]['#hardness_weighted_questions']
                        so_far_individual_performance[member] = {
                            'correct_rate_so_far': correct_rate,
                            'hardness_weighted_correct_rate_so_far': hardness_weighted_correct_rate}
                    self.individual_performance_rates[team_id].append(
                        so_far_individual_performance)

    def generate_dataset(self):
        X = []
        y = []
        for team_id, team_log in self.data.items():
            if team_id in self.networks:
                print("In generate_dataset: processing team", team_id, '...')

                # First influence matrix:
                first_index = 0
                while first_index < len(self.networks[team_id]):
                    influence_matrix = np.matrix(
                        team_log.member_influences[first_index])
                    if self.skip_matrices_not_completely_from_members and np.sum(team_log.member_influences_from_data[first_index]) != 16:
                        print('E1: Index: {} was skipped.'.format(first_index))
                        first_index += 1
                        continue
                    normalized_influence_matrix = utils.shuffle_matrix_in_given_order(
                        matrix=influence_matrix,
                        order=np.argsort(team_log.members)) / 100
                    first_row_stochastic_normalized_influence_matrix = np.matrix(
                        utils.make_matrix_row_stochastic(normalized_influence_matrix))
                    previous_row_stochastic_normalized_influence_matrix = first_row_stochastic_normalized_influence_matrix.copy()
                    break

                # Average of previous influence matrices:
                previous_influence_matrices_cnt = 1
                # CHECK IF THIS IS NOTHING
                average_of_previous_influence_matrices = first_row_stochastic_normalized_influence_matrix.copy()
                for index in range(first_index + 1, len(self.networks[team_id])):
                    influence_matrix = np.matrix(
                        team_log.member_influences[index])
                    if self.skip_matrices_not_completely_from_members and np.sum(team_log.member_influences_from_data[index]) != 16:
                        print('E2: Index: {} was skipped.'.format(index))
                        continue

                    # Individual performance:
                    individual_performance = np.zeros(4)
                    individual_performance_hardness_weighted = np.zeros(4)
                    perf_rates = self.individual_performance_rates[team_id][index]
                    for i, member in enumerate(sorted(team_log.members)):
                        individual_performance[i] = perf_rates[member]['correct_rate_so_far']
                        individual_performance_hardness_weighted[i] = perf_rates[
                            member]['hardness_weighted_correct_rate_so_far']

                    # Networks:
                    network = self.networks[team_id][index]

                    # Contents:
                    contents_embedding = self.contents_embeddings[team_id][index]

                    # Average of previous influence matrices:
                    normalized_influence_matrix = utils.shuffle_matrix_in_given_order(
                        matrix=influence_matrix,
                        order=np.argsort(team_log.members)) / 100
                    row_stochastic_normalized_influence_matrix = np.matrix(
                        utils.make_matrix_row_stochastic(normalized_influence_matrix))

                    # Multi-class classification (who is (are) the most influential individual(s)):
                    most_influentials = utils.most_influential_on_others(
                        influence_matrix=row_stochastic_normalized_influence_matrix,
                        remove_self_influence=True)

                    # Combining all features together:
                    y.append({
                        'influence_matrix': row_stochastic_normalized_influence_matrix,
                        'most_influentials': most_influentials})
                    X.append({
                        'individual_performance': individual_performance,
                        'individual_performance_hardness_weighted': individual_performance_hardness_weighted,
                        'content_embedding_matrix': contents_embedding,
                        'first_influence_matrix': first_row_stochastic_normalized_influence_matrix,
                        'previous_influence_matrix': previous_row_stochastic_normalized_influence_matrix,
                        'average_of_previous_influence_matrices': average_of_previous_influence_matrices / previous_influence_matrices_cnt,
                        'reply_duration': nx.adj_matrix(network['reply_duration']).todense(),
                        'sentiment': nx.adj_matrix(network['sentiment']).todense(),
                        'emotion_arousal': nx.adj_matrix(network['emotion_arousal']).todense(),
                        'emotion_dominance': nx.adj_matrix(network['emotion_dominance']).todense(),
                        'emotion_valence': nx.adj_matrix(network['emotion_valence']).todense()})
                    previous_row_stochastic_normalized_influence_matrix = row_stochastic_normalized_influence_matrix.copy()
                    average_of_previous_influence_matrices += row_stochastic_normalized_influence_matrix
                    previous_influence_matrices_cnt += 1

        self.supervised_data = {'X': X, 'y': y}
