# Omid55
# Start date:     26 Sept 2019
# Modified date:  2 Nov 2019
# Author:   Omid Askarisichani, Koa Sato
# Email:    omid55@cs.ucsb.edu, kekoa_sato@ucsb.edu
#
# Module for processing the POGS Jeopardy-like log data.
# Example:
#

from __future__ import division, print_function, absolute_import, unicode_literals

import attr
import json
import re
from enum import Enum, unique
import numpy as np
import pandas as pd
from typing import Text
import copy


class Error(Exception):
    """Base Error class for this module."""


class NotFoundFileError(Error):
    """Error class when a file is not found."""


class EventLogsNotLoadedError(Error):
    """Error class when event logs haven't been loaded in the constructor."""


@unique
class Category(Enum):
    SCIENCE_AND_TECHNOLOGY = 1
    HISTORY_AND_MYTHOLOGY = 2
    LITERATURE_AND_MEDIA = 3

    @staticmethod
    def parse(category_str: Text) -> 'Category':
        str2category_map = {
            'Science and Technology': Category.SCIENCE_AND_TECHNOLOGY,
            'History and Mythology': Category.HISTORY_AND_MYTHOLOGY,
            'Literature and Media': Category.LITERATURE_AND_MEDIA}
        if category_str not in str2category_map:
            raise ValueError(
                'The input category was not found. It was entered: {}'.format(
                    category_str))
        return str2category_map[category_str]


@unique
class Level(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3

    @staticmethod
    def parse(level_str: Text) -> 'Level':
        str2level_map = {
            'Easy': Level.EASY,
            'Medium': Level.MEDIUM,
            'Hard': Level.HARD}
        if level_str not in str2level_map:
            raise ValueError(
                'The input level was not found. It was entered: {}'.format(
                    level_str))
        return str2level_map[level_str]


@attr.s
class JeopardyQuestion(object):
    """Jeopardy-like question including category, choices, right answer, etc."""
    id = attr.ib(type=int)
    question_content = attr.ib(type=str)
    answer = attr.ib(type=str)
    choices = attr.ib(factory=list)
    category = attr.ib(factory=Category)
    level = attr.ib(factory=Level)


@attr.s
class JeopardyInfoOptions(object):
    """Options about the executed game."""
    num_of_questions = attr.ib(default=45)
    num_of_agents = attr.ib(default=4)
    num_of_team_members = attr.ib(default=4)
    num_of_influence_reports = attr.ib(default=9)
    correct_points = attr.ib(default=4)
    wrong_points = attr.ib(default=-1)
    using_agent_points = attr.ib(default=-1)
    questions = attr.ib(factory=list)

@attr.s
class MachineInfo():
    """Info about a machine's usage per question"""
    used = attr.ib(default=False)
    probability = attr.ib(default=-1)
    user = attr.ib(default=-1)
    answer_given = attr.ib(default="")


class TeamLogProcessor(object):
    """Processes the logs of one team who played POGS Jeopardy game.

        Usage:
            loader = TeamLogProcessor(
                team_id=1,
                logs_directory_path='/home/omid/Datasets/Jeopardy',
                game_info=JeopardyInfoOptions())

        Properties:
            team_id: The id of the existing team in this object.
            game_info: Jeopardy-like game information.
            team_event_logs: All event logs for this team.
            messages: The communication messages among members.
            members: List of members in the correct order.
    """
    def __init__(self,
                 team_id: int,
                 logs_directory_path: Text,
                 game_info: JeopardyInfoOptions = JeopardyInfoOptions()):
        self.team_id = team_id
        self.game_info = game_info
        self._load_all_files(logs_directory_path)

    def _load_all_files(self, logs_directory_path: Text) -> None:
        """Load all files pipeline.

        This function only calls all other unit tested functions and does not
        need testing.
        """
        self.logs_directory_path = logs_directory_path
        self._load_game_questions(
            file_path=logs_directory_path + '/jeopardy.json')
        self._load_this_team_event_logs(
            logs_file_path=logs_directory_path + '/event_log.csv',
            team_has_subject_file_path=logs_directory_path
                + '/team_has_subject.csv')
        self._load_messages()
        self._load_answers_chosen()
        self._load_machine_usage_info()
        self._preload_data(logs_directory_path)
        self._define_team_member_order(logs_directory_path)
        self._load_ratings()
        self._load_accumulated_score()
        self._load_survey()
        # self._old_load_all(logs_directory_path, self.team_id)  ## DELETE.

    def _load_this_team_event_logs(self,
                                   logs_file_path:Text,
                                   team_has_subject_file_path: Text) -> None:
        event_log = pd.read_csv(
            logs_file_path,
            sep=',',
            quotechar='|',
            names=['id', 'event_type', 'event_content', 'timestamp', 
                   'completed_task_id', 'sender_subject_id',
                   'receiver_subject_id', 'session_id', 'sender', 'receiver',
                   'extra_data'])
        team_subjects = pd.read_csv(
            team_has_subject_file_path,
            sep=',',
            quotechar='|',
            names=['id', 'team_id', 'sender_subject_id']).drop('id', 1)
        # event_log = pd.read_csv(logs_file_path, sep=',',quotechar="|", names=["id","event_type","event_content","timestamp","completed_task_id","sender_subject_id","receiver_subject_id","session_id","sender","receiver","extra_data"])
        # team_subjects = pd.read_csv(team_has_subject_file_path,sep=',',quotechar="|",names=["id","team_id","sender_subject_id"]).drop('id',1)
        event_log['sender_subject_id'] = pd.to_numeric(
            event_log['sender_subject_id'])
        event_log_with_team = pd.merge(
            event_log, team_subjects, on='sender_subject_id', how='left')
        self.team_event_logs = event_log_with_team[
            (event_log_with_team['team_id'] == self.team_id)]
        if len(self.team_event_logs) == 0:
            raise EventLogsNotLoadedError(
                'Logs for team_id={} was not found.'.format(self.team_id))

    def _load_game_questions(self, file_path: Text) -> None:
        """Loads every question, choices and the right answer.
        """
        self.game_info.questions = {}
        with open(file_path, 'r') as f:
            question_list = json.load(f)
            self.game_info.num_of_questions = len(question_list)
            for question in question_list:
                self.game_info.questions[question['ID']] = JeopardyQuestion(
                    id=question['ID'],
                    question_content=question['question'],
                    answer=question['Answer'],
                    choices=question['value'],
                    category=Category.parse(question['Category']),
                    level=Level.parse(question['Level']))

    def _load_messages(self) -> None:
        """Loads the communication messages for the current team.

        Args:
            None.

        Returns:
            None.

        Raises:
            EventLogsNotLoadedError: If the constructor has not been loaded the
                event logs data yet.
        """
        if len(self.team_event_logs) == 0:
            raise EventLogsNotLoadedError(
                'Please first run constructor of TeamLogProcessor.')
        indices = [0] + list(
            np.where(self.team_event_logs.extra_data == 'SubmitButtonField')[0])
        begin_index = 0
        end_index = 1
        def extract_message(message_content):
            return message_content.split('"message":"')[1].split('"')[0]
        self.messages = []
        while end_index < len(indices):
            if indices[end_index] - indices[begin_index] > 4:
                df = self.team_event_logs.iloc[
                    indices[begin_index] + 1: indices[end_index]]
                df = df[df.event_type == 'COMMUNICATION_MESSAGE']
                df.event_content = df.event_content.apply(extract_message)
                df = df[df.event_content != '']
                self.messages.append(df)
            begin_index = end_index
            end_index += 1

    def _get_last_individual_answers(self, individual_answers_chosen):
        last_answers = {}
        for index, row in individual_answers_chosen.iterrows():
            answer = row["event_content"].split(',')[0].replace('"', '')
            last_answers[row["sender_subject_id"]] = answer
        return last_answers

    def _get_last_group_answers(self, group_answers_chosen, last_answers):
        for index, row in group_answers_chosen.iterrows():
            answer = row["event_content"].split(',')[0].replace('"', '')
            last_answers[row["sender_subject_id"]] = answer
        return last_answers

    def _load_answers_chosen(self) -> None:
        """Loads the choices of each person for their initial and final answer"""
        indices = [0] + list(
            np.where(self.team_event_logs.extra_data == 'SubmitButtonField')[0])
        if len(indices) == 1:
            raise EventLogsNotLoadedError(
                'No answer were found for team {}.'.format(self.team_id))
        begin_index = 0
        end_index = 1

        def extract_answer_and_question(event_content):
            event_content_string = event_content[1:-1]
            event_content_array = event_content_string.split('||')
            answer_info = event_content_array[0].split(':')
            answer = answer_info[1]
            question_info = event_content_array[1].split(':')
            question_number = question_info[1]
            return answer + "," + question_number

        individual_answers_chosen_list = []
        group_answers_chosen = []
        self.question_order = []

        while end_index < len(indices):
            if indices[end_index] - indices[begin_index] > 4:
                df = self.team_event_logs.iloc[
                    indices[begin_index] + 1: indices[end_index]]
                df = df[df.extra_data == 'IndividualResponse']
                df.event_content = df.event_content.apply(
                    extract_answer_and_question)
                individual_answers_chosen_list.append(df)

                df = self.team_event_logs.iloc[
                    indices[begin_index] + 1: indices[end_index]]
                df = df[df.extra_data == 'GroupRadioResponse']
                df.event_content = df.event_content.apply(extract_answer_and_question)
                df = df[df.event_content != '']
                group_answers_chosen.append(df)

            begin_index = end_index
            end_index += 1

        self.individual_answers_chosen = {}
        self.group_answers_chosen = {}
        # self._set_team_members(individual_answers_chosen_list[0])
        for index in range(len(individual_answers_chosen_list)):
            event_content = str(individual_answers_chosen_list[index].event_content)
            question_number = int(float(event_content.split("\n")[0].split(",")[1]))

            self.question_order.append(question_number)
            last_answers = self._get_last_individual_answers(individual_answers_chosen_list[index])
            self.individual_answers_chosen[question_number] = last_answers
            last_group_answers = copy.deepcopy(last_answers)
            last_group_answers = self._get_last_group_answers(group_answers_chosen[index], last_group_answers)
            self.group_answers_chosen[question_number] = last_group_answers


    def _load_machine_usage_info(self) -> None:
        """Loads whether the machine was used for every question."""
        indices = [0] + list(
            np.where(self.team_event_logs.extra_data == 'SubmitButtonField')[0])
        begin_index = 0
        end_index = 1
        def extract_machine_info(event_content):
            event_content_string = event_content[1:-1]
            event_content_array = event_content_string.split('||')
            machine_info = event_content_array[0].replace('"', '').split(':')
            answer = machine_info[2].split("_")[0]
            probability = float(machine_info[2].split("_")[1])
            question_info = event_content_array[1].split(':')
            question_number = int(float(question_info[1]))

            return probability, answer, question_number

        def get_info_as_list(event_content):
            event_content = event_content[event_content.find("(") + 1:-1]
            event_content_list = event_content.split(",")
            probability = float(event_content_list[0])
            answer = event_content_list[1].strip()
            question_number = int(event_content_list[2])
            return [probability, answer, question_number]

        def extract_question_number(event_content):
            event_content_string = event_content[1:-1]
            event_content_array = event_content_string.split('||')
            question_info = event_content_array[1].split(':')
            question_number = question_info[1]
            return question_number

        self.machine_usage_info = {}
        while end_index < len(indices):
            if indices[end_index] - indices[begin_index] > 4:
                df = self.team_event_logs.iloc[
                    indices[begin_index] + 1: indices[end_index]]
                df = df[df.extra_data == 'AskedMachine']

                if (not df.empty):
                    df.event_content = df.event_content.apply(extract_machine_info)
                    info = get_info_as_list(df.event_content.to_string().split('\n')[0])
                    user = int(df.iloc[0]["sender_subject_id"])
                    machine_info = MachineInfo(
                        used=True,
                        probability=info[0],
                        user=user,
                        answer_given=info[1])
                    self.machine_usage_info[info[2]] = machine_info
                else:
                    df = self.team_event_logs.iloc[
                    indices[begin_index] + 1: indices[end_index]]
                    df = df[df.extra_data == 'IndividualResponse']
                    df.event_content = df.event_content.apply(extract_question_number)
                    question_number = int(float(df.iloc[0]["event_content"]))
                    self.machine_usage_info[question_number] = MachineInfo()

            begin_index = end_index
            end_index += 1

    # def _create_team_member_mapping(self, df) -> None:
    #     self.team_member_mapping = {}
    #     for index in range(len(df)):
    #         sender_subject_id = int(str(df.sender_subject_id.iloc[index]))
    #         session_id = str(df[df.columns[8]].iloc[index])
    #         if session_id not in self.team_member_mapping:
    #             self.team_member_mapping[session_id] = sender_subject_id

    # def _add_to_team_member_mapping(self, sender) -> None:
    #     subject = pd.read_csv(
    #         self.logs_directory_path + 'subject.csv',
    #         sep=',',
    #         quotechar='|',
    #         names=['sender_subject_id', 'sender', 'sender_dup', 'group',
    #                'empty'])

    #     row = subject.loc[subject['sender'] == sender]
    #     item = row.iloc[0][0]
    #     self.team_member_mapping[sender] = item


    def _preload_data(self, directory) -> None:
        # Preloading of the data
        self.event_log = pd.read_csv(directory+"/event_log.csv", sep=',',quotechar="|", names=["id","event_type","event_content","timestamp","completed_task_id","sender_subject_id","receiver_subject_id","session_id","sender","receiver","extra_data"])
        self.team_subjects = pd.read_csv(directory+"/team_has_subject.csv",sep=',',quotechar="|",names=["id","teamId","sender_subject_id"]).drop('id',1)
        event_log_no_message =  self.event_log[(self.event_log['event_type'] == "TASK_ATTRIBUTE")]
        event_log_no_message["sender_subject_id"] = pd.to_numeric(event_log_no_message["sender_subject_id"])

        event_log_with_team = pd.merge(event_log_no_message, self.team_subjects, on='sender_subject_id', how='left')
        event_log_task_attribute = event_log_with_team[(event_log_with_team['event_type'] == "TASK_ATTRIBUTE") & (event_log_with_team['teamId'] == self.team_id)]
        #Extract data from event_content column
        new_event_content = pd.DataFrame(
            index=np.arange(0, len(event_log_task_attribute)),
            columns=("id","stringValue", "questionNumber","questionScore","attributeName"))
        self.questionNumbers = list()

        for i in range(len(event_log_task_attribute)):
            new_event_content.id[i] = event_log_task_attribute.iloc[i]["id"]
            new_event_content.stringValue[i] = event_log_task_attribute.iloc[i]["event_content"].split("||")[0].split(":")[1].replace('"', '')
            new_event_content.questionNumber[i] = event_log_task_attribute.iloc[i]["event_content"].split("||")[1].split(":")[1]
            if new_event_content.questionNumber[i] not in self.questionNumbers:
                self.questionNumbers.append(new_event_content.questionNumber[i])
            new_event_content.questionScore[i] = event_log_task_attribute.iloc[i]["event_content"].split("||")[3].split(":")[1]
            new_event_content.attributeName[i] =event_log_task_attribute.iloc[i]["event_content"].split("||")[2].split(":")[1]

        self.questionNumbers = self.questionNumbers[1 :]
        self.event_log_with_all_data = pd.merge(event_log_task_attribute,new_event_content,on='id', how ='left')


    def _define_team_member_order(self, directory) -> None:
        # Define teammember order
        subjects = pd.read_csv(directory+"/subject.csv", sep=',',quotechar="|", names=["sender_subject_id","externalId","displayName","sessionId","previousSessionSubject"])
        team_with_subject_details = pd.merge(self.team_subjects, subjects, on='sender_subject_id', how='left')
        self.team_member = team_with_subject_details[(team_with_subject_details['teamId'] == self.team_id)]['displayName']
        self.team_size = len(self.team_member)
        self.team_array = []
        for i in range(self.team_size):
            self.team_array.append(self.team_member.iloc[i])
        self.members = []
        for member in self.team_array:
            self.members.append(
                list(subjects[subjects['displayName'] == member]['sender_subject_id'])[0])

    def _extract_and_fill_missing_values(self, temp, aR, mI, aR_from_data, mI_from_data):
        for j in range(0, self.team_size):
            agent_from_data = True
            member_from_data = True
            # Fill missing values
            xy = re.findall(r'Ratings(.*?) Member', temp)[0].split("+")[j].split("=")[1]
            if(xy==''):
                xy = '0.0'
                agent_from_data = False
            yz= temp.replace('"', '')[temp.index("Influences ")+10:].split("+")[j].split("=")[1]
            if(yz == ''):
                yz = '25'
                member_from_data = False
            aR.append(float(xy))
            mI.append(int(round(float(yz))))
            aR_from_data.append(agent_from_data)
            mI_from_data.append(member_from_data)

    def _add_values_for_missing_line(self, count, missing_members, a_ratings,
        m_influences, a_ratings_from_data, m_influences_from_data):
        for member in missing_members:
            aR = list()
            mI = list()
            aR_from_data = list()
            mI_from_data = list()
            idx = self.team_array.index(member)
            for j in range(0, self.team_size):
                aR.append(0.0)
                mI.append(25)
                aR_from_data.append(False)
                mI_from_data.append(False)
            a_ratings[idx] = aR
            m_influences[idx] = mI
            a_ratings_from_data[idx] = aR_from_data
            m_influences_from_data[idx] = mI_from_data
            count += 1
        return count

    def _load_ratings(self) -> None:
        self.agent_ratings = list()
        self.member_influences = list()
        self.agent_ratings_from_data = list()
        self.member_influences_from_data = list()
        m_influences = [0 for i in range(self.team_size)]
        a_ratings = [0 for i in range(self.team_size)]
        m_influences_from_data = [False for i in range(self.team_size)]
        a_ratings_from_data = [False for i in range(self.team_size)]
        count = 0
        influence_matrices = self.event_log_with_all_data[(self.event_log_with_all_data['extra_data'] == "InfluenceMatrix")]
        influence_matrix_without_undefined = influence_matrices[~influence_matrices['stringValue'].str.contains("undefined")]
        final_influences = influence_matrix_without_undefined.groupby(['questionScore', 'sender'], as_index=False, sort=False).last()

        processed_members = []
        current_question_score = None

        # Loop that extracts values and fills in missing ones for all
        # InfluenceMatrix entries
        for i in range(len(final_influences)):
            count +=1
            aR = list()
            mI = list()
            aR_from_data = list()
            mI_from_data = list()
            idx = self.team_array.index(final_influences.iloc[i]['sender'])
            processed_members.append(final_influences.iloc[i]['sender'])
            current_question_score = int(final_influences.iloc[i]['questionScore'])

            a_ratings[idx]=aR
            m_influences[idx]=mI
            a_ratings_from_data[idx] = aR_from_data
            m_influences_from_data[idx] = mI_from_data
            temp = final_influences.iloc[i]['stringValue']
            self._extract_and_fill_missing_values(temp, aR, mI, aR_from_data, mI_from_data)

            # Need to check if there is a next influence matrix line or
            # if the next line belongs to a different round
            # If so, check if we missed a team member's answers (missing line)
            # and fill in values
            if (i + 2 > len(final_influences) or
                int(final_influences.iloc[i + 1]['questionScore']) != current_question_score):
                missing_members = np.setdiff1d(self.team_array, processed_members)

                count = self._add_values_for_missing_line(count, missing_members,
                a_ratings, m_influences, a_ratings_from_data, m_influences_from_data)
                processed_members = []

            # If we saw everyone's answers, then add their responses to the
            # influence matrix data structure along with whether the answers
            # were from data or not.
            if (count == self.team_size):
                self.member_influences.append(m_influences)
                self.agent_ratings.append(a_ratings)
                self.member_influences_from_data.append(m_influences_from_data)
                self.agent_ratings_from_data.append(a_ratings_from_data)

                m_influences = [0 for i in range(self.team_size)]
                a_ratings = [0 for i in range(self.team_size)]
                m_influences_from_data = [False for i in range(self.team_size)]
                a_ratings_from_data = [False for i in range(self.team_size)]

                count = 0

    def _load_accumulated_score(self) -> None:
        """Loads the accumulated score per question"""
        self.score = {}
        self.accumulated_score = {}
        self.accumulated_score[0] = 0
        index = 1
        for i in self.question_order:

            score_earned = 0
            final_answer_chosen = None
            if len(set(self.group_answers_chosen[i].values())) == 1:
                final_answer_chosen = self.group_answers_chosen[i][list(self.group_answers_chosen[i].keys())[0]]

            if i not in self.game_info.questions:
                print(
                    'Warning: question {} was not found in the game info.'.format(i))
                continue

            answer = self.game_info.questions[i].answer

            if (final_answer_chosen == answer):
                score_earned = score_earned + 4
            else:
                score_earned = score_earned - 1

            if self.machine_usage_info[i].used:
                score_earned = score_earned - 1

            self.score[i] = score_earned
            self.accumulated_score[index] = self.accumulated_score[index - 1] + score_earned
            index = index + 1

    def _load_survey(self) -> None:
        pre_experiment_data = self.event_log_with_all_data[self.event_log_with_all_data['extra_data'] == "RadioField"]

        self.pre_experiment_rating = []
        for i in range(len(self.team_array)):
            survey_dict = { 0: -1, 1: -1, 2: -1}
            for row in range (len(pre_experiment_data) - 1, -1, -1):
                if (pre_experiment_data.iloc[row]['sender'] == self.team_member.iloc[i]):
                    current_frame = pre_experiment_data.iloc[row]
                    if (current_frame['attributeName'] == "\"surveyAnswer0\"" and
                        survey_dict[0] == -1):
                        survey_dict[0] = float(current_frame['stringValue'][0:1])
                    elif(current_frame['attributeName'] == "\"surveyAnswer1\"" and
                        survey_dict[1] == -1):
                        survey_dict[1] = float(current_frame['stringValue'][0:1])
                    elif(current_frame['attributeName'] == "\"surveyAnswer2\"" and
                        survey_dict[2] == -1):
                        survey_dict[2] = float(current_frame['stringValue'][0:1])
            self.pre_experiment_rating.append(survey_dict)

    def _old_load_all(self, directory, teamId):
            #Constants
        self.numQuestions = 45
        self.trainingSetSize = 30
        self.testSetSize = 15
        self.numAgents = 4

        self.numCentralityReports = 9

        self.c = 4
        self.e = -1
        self.z = -1

#         Other Parameters
        self.influenceMatrixIndex = 0
        self.machineUseCount = [-1, -1, -1, -1]
        self.firstMachineUsage = [-1, -1, -1, -1]
        
        # Preloading of the data
        eventLog = pd.read_csv(directory+"/event_log.csv", sep=',',quotechar="|", names=["id","event_type","event_content","timestamp","completed_task_id","sender_subject_id","receiver_subject_id","session_id","sender","receiver","extra_data"])
        teamSubjects = pd.read_csv(directory+"/team_has_subject.csv",sep=',',quotechar="|",names=["id","teamId","sender_subject_id"]).drop('id',1)
        elNoMessage =  eventLog[(eventLog['event_type'] == "TASK_ATTRIBUTE")]
        elNoMessage["sender_subject_id"] = pd.to_numeric(elNoMessage["sender_subject_id"])
        
        eventLogWithTeam = pd.merge(elNoMessage, teamSubjects, on='sender_subject_id', how='left')
        eventLogTaskAttribute = eventLogWithTeam[(eventLogWithTeam['event_type'] == "TASK_ATTRIBUTE") & (eventLogWithTeam['teamId'] == teamId)]
        #Extract data from event_content column
        newEventContent = pd.DataFrame(
            index=np.arange(0, len(eventLogTaskAttribute)),
            columns=("id","stringValue", "questionNumber","questionScore","attributeName"))
        self.questionNumbers = list()

        for i in range(len(eventLogTaskAttribute)):
            newEventContent.id[i] = eventLogTaskAttribute.iloc[i]["id"]
            newEventContent.stringValue[i] = eventLogTaskAttribute.iloc[i]["event_content"].split("||")[0].split(":")[1].replace('"', '')
            newEventContent.questionNumber[i] = eventLogTaskAttribute.iloc[i]["event_content"].split("||")[1].split(":")[1]
            if newEventContent.questionNumber[i] not in self.questionNumbers:
                self.questionNumbers.append(newEventContent.questionNumber[i])
            newEventContent.questionScore[i] = eventLogTaskAttribute.iloc[i]["event_content"].split("||")[3].split(":")[1]
            newEventContent.attributeName[i] =eventLogTaskAttribute.iloc[i]["event_content"].split("||")[2].split(":")[1]
            
        self.questionNumbers = self.questionNumbers[1 :]
        eventLogWithAllData = pd.merge(eventLogTaskAttribute,newEventContent,on='id', how ='left')
                
        self.machineAsked = eventLogWithAllData[eventLogWithAllData['extra_data'] == "AskedMachine"]
        self.machineAskedQuestions = list()
        for i in range(len(self.machineAsked)):
            self.machineAskedQuestions.append(int(float(self.machineAsked.iloc[i]['questionNumber'])))
        
        # Load correct answers
        with open(directory+"/jeopardy.json") as json_data:
                d = json.load(json_data)
        self.correctAnswers = list()
        self.options = list()
        
        for i in range(len(self.questionNumbers)):
            self.correctAnswers.append(d[int(float(self.questionNumbers[i]))-1]['Answer'])
            self.options.append(d[int(float(self.questionNumbers[i]))-1]['value'])
        
        allIndividualResponses = eventLogWithAllData[eventLogWithAllData['extra_data'] == "IndividualResponse"]
        self.lastIndividualResponsesbyQNo = allIndividualResponses.groupby(['sender', 'questionNumber'], as_index=False, sort=False).last()
        
        # Compute the group answer of the team per question
        submissions = eventLogWithAllData[(eventLogWithAllData['extra_data'] == "IndividualResponse") | (eventLogWithAllData['extra_data'] == "GroupRadioResponse") ]
        individualAnswersPerQuestion = submissions.groupby(["questionNumber","sender_subject_id"], as_index=False, sort=False).tail(1)
        
        self.groupSubmission = pd.DataFrame(index=np.arange(0, len(self.questionNumbers)), columns=("questionNumber","groupAnswer"))
        for i in range(len(self.questionNumbers)):
            ans = ""
            consensusReached = True
            for j in range(len(individualAnswersPerQuestion)):
                if (individualAnswersPerQuestion.iloc[j].loc["questionNumber"] == self.questionNumbers[i]):
                    if not ans:
                        ans = individualAnswersPerQuestion.iloc[j].loc["stringValue"]
                    elif (ans != individualAnswersPerQuestion.iloc[j].loc["stringValue"]):
                        consensusReached = False
                        break
                        
            self.groupSubmission.questionNumber[i] = self.questionNumbers[i]
            if (consensusReached):
                self.groupSubmission.groupAnswer[i] = ans
            else:
                self.groupSubmission.groupAnswer[i] = "Consensus Not Reached"
             
        # Define teammember order
        subjects = pd.read_csv(directory+"/subject.csv", sep=',',quotechar="|", names=["sender_subject_id","externalId","displayName","sessionId","previousSessionSubject"])
        teamWithSujectDetails = pd.merge(teamSubjects, subjects, on='sender_subject_id', how='left')
        self.teamMember = teamWithSujectDetails[(teamWithSujectDetails['teamId'] == teamId)]['displayName']        
        self.teamSize = len(self.teamMember)
        self.teamArray = list()
        
        for i in range(self.teamSize):
            self.teamArray.append(self.teamMember.iloc[i])
        
        #         Pre-experiment Survey
        preExperimentData = eventLogWithAllData[eventLogWithAllData['extra_data'] == "RadioField"]
        self.preExperimentRating = list()
        for i in range(0,self.teamSize):
            self.preExperimentRating.append(0)
            if len(preExperimentData[(preExperimentData['sender'] == self.teamMember.iloc[i]) & (preExperimentData['attributeName'] == "\"surveyAnswer0\"")])>0:
                self.preExperimentRating[-1]+=(float(preExperimentData[(preExperimentData['sender'] == self.teamMember.iloc[i]) & (preExperimentData['attributeName'] == "\"surveyAnswer0\"")]['stringValue'].iloc[0][0:1]))
            if len(preExperimentData[(preExperimentData['sender'] == self.teamMember.iloc[i]) & (preExperimentData['attributeName'] == "\"surveyAnswer1\"")]) >0:
                self.preExperimentRating[-1]+=(float(preExperimentData[(preExperimentData['sender'] == self.teamMember.iloc[i]) & (preExperimentData['attributeName'] == "\"surveyAnswer1\"")]['stringValue'].iloc[0][0:1]))
            if len(preExperimentData[(preExperimentData['sender'] == self.teamMember.iloc[i]) & (preExperimentData['attributeName'] == "\"surveyAnswer2\"")])>0:
                self.preExperimentRating[-1]+=(float(preExperimentData[(preExperimentData['sender'] == self.teamMember.iloc[i]) & (preExperimentData['attributeName'] == "\"surveyAnswer2\"")]['stringValue'].iloc[0][0:1]))
            self.preExperimentRating[-1]/=15
        
        # Extracting Machine Usage Information
        self.machineUsed = np.array([False, False, False, False] * self.numQuestions).reshape((self.numQuestions, 4))             
        for i in range(len(self.questionNumbers)):
            if int(float(self.questionNumbers[i])) in self.machineAskedQuestions:
                indxM = self.machineAskedQuestions.index(int(float(self.questionNumbers[i])))            
                k = self.teamArray.index(self.machineAsked['sender'].iloc[indxM])
                self.machineUsed[i][int(k)] = True
        
        self.teamScore = list()           
        self.teamScore.append(0)
        for i in range(len(self.questionNumbers)):
            if self.groupSubmission.groupAnswer[i]!=self.correctAnswers[i]:
                self.teamScore[i]+=self.z
            else:
                self.teamScore[i]+=self.c    
            if len(np.where(self.machineUsed[i] == True)[0])!=0:
                self.teamScore[i]+=self.e
            self.teamScore.append(self.teamScore[i])
        self.teamScore = self.teamScore[:-1]
                
#         Extract Influence Matrices
        self.agentRatings = list()
        self.memberInfluences = list()
        mInfluences = [0 for i in range(self.teamSize)]
        aRatings = [0 for i in range(self.teamSize)]
        count = 0 
        influenceMatrices = eventLogWithAllData[(eventLogWithAllData['extra_data'] == "InfluenceMatrix")]  
        influenceMatrixWithoutUndefined = influenceMatrices[~influenceMatrices['stringValue'].str.contains("undefined")]
        finalInfluences = influenceMatrixWithoutUndefined.groupby(['questionScore', 'sender'], as_index=False, sort=False).last()
        
        for i in range(len(finalInfluences)):
            count +=1 
            aR = list()
            mI = list() 
            idx = self.teamArray.index(finalInfluences.iloc[i]['sender'])
            for j in range(0, self.teamSize):
                temp = finalInfluences.iloc[i]['stringValue']
#                 Fill missing values
                xy = re.findall(r'Ratings(.*?) Member', temp)[0].split("+")[j].split("=")[1]
                if(xy==''):
                    xy = '0.5'
                yz= temp.replace('"', '')[temp.index("Influences ")+10:].split("+")[j].split("=")[1]
                if(yz == ''):
                    yz = '25'
                aR.append(float(xy))
                mI.append(int(round(float(yz))))
            aRatings[idx]=aR
            mInfluences[idx]=mI 
            if(count%self.teamSize == 0):
                self.memberInfluences.append(mInfluences)
                mInfluences = [0 for i in range(self.teamSize)]
                self.agentRatings.append(aRatings)
                aRatings = [0 for i in range(self.teamSize)]
        
        # # Hyperparameters for expected performance (Humans and Agents) - TODO
        # self.alphas = [1,1,1,1,1,1,1,1]
        # self.betas = np.ones(8, dtype = int)

        # #vector c
        # self.centralities = [[] for _ in range(self.numQuestions)]

        self.actionTaken = list()
        for i in range(len(self.questionNumbers)):
            if self.groupSubmission.groupAnswer[i] == "Consensus Not Reached":
                self.actionTaken.append(-1)
            elif int(float(self.questionNumbers[i])) in self.machineAskedQuestions:
                self.actionTaken.append(self.teamSize + np.where(self.machineUsed[i] == True)[0][0])
            else:
                self.actionTaken.append(self.options[i].index(self.groupSubmission.groupAnswer[i]))