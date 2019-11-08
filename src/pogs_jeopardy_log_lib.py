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
            logs_file_path=logs_directory_path + 'event_log.csv',
            team_has_subject_file_path=logs_directory_path
                + 'team_has_subject.csv')
        self._load_messages()
        self._load_answers_chosen()
        self._load_machine_usage_info()
        self._load_ratings()
        self._load_accumulated_score()
        #self._old_load_all(logs_directory_path, self.team_id)  ## DELETE.

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
        with open(file_path, 'r') as f:
            question_list = json.load(f)
            self.game_info.num_of_questions = len(question_list)
            for question in question_list:
                self.game_info.questions.append(
                    JeopardyQuestion(
                        id=question['ID'],
                        question_content=question['question'],
                        answer=question['Answer'],
                        choices=question['value'],
                        category=Category.parse(question['Category']),
                        level=Level.parse(question['Level'])))

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

    def _set_team_members(self, individual_answers_chosen):
        self.members = []
        for index, row in individual_answers_chosen.iterrows():
            if ((row["sender_subject_id"]) not in self.members):
                self.members.append(row["sender_subject_id"])

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
        """Loads the answer choices of each person for their initial and final answer"""
        indices = [0] + list(
            np.where(self.team_event_logs.extra_data == 'SubmitButtonField')[0])
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

        individual_answers_chosen = []
        group_answers_chosen = []
        self.question_order = []

        while end_index < len(indices):
            if indices[end_index] - indices[begin_index] > 4:
                df = self.team_event_logs.iloc[
                    indices[begin_index] + 1: indices[end_index]]
                df = df[df.extra_data == 'IndividualResponse']
                df.event_content = df.event_content.apply(extract_answer_and_question)
                individual_answers_chosen.append(df)

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
        self._set_team_members(individual_answers_chosen[0])
        for index in range(len(individual_answers_chosen)):
            event_content = str(individual_answers_chosen[index].event_content)
            question_number = int(float(event_content.split("\n")[0].split(",")[1]))

            self.question_order.append(question_number)
            last_answers = self._get_last_individual_answers(individual_answers_chosen[index])
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
                    info = get_info_as_list(df.event_content.to_string())
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

    def _create_team_member_mapping(self, df) -> None:
        self.team_member_mapping = {}
        for index in range(len(df)):
            sender_subject_id = int(str(df.sender_subject_id.iloc[index]))
            session_id = str(df[df.columns[8]].iloc[index])
            if session_id not in self.team_member_mapping:
                self.team_member_mapping[session_id] = sender_subject_id

    def _add_to_team_member_mapping(self, sender) -> None:
        subject = pd.read_csv(
            self.logs_directory_path + 'subject.csv',
            sep=',',
            quotechar='|',
            names=['sender_subject_id', 'sender', 'sender_dup', 'group',
                   'empty'])

        row = subject.loc[subject['sender'] == sender]
        item = row.iloc[0][0]
        self.team_member_mapping[sender] = item

    def _load_ratings(self) -> None:
        """Loads the agent and member ratings."""
        indices = [0] + list(
            np.where(self.team_event_logs.extra_data == 'SubmitButtonField')[0])
        begin_index = 0
        end_index = 1

        def parse_event_content(event_content, team_member_id):
            agent_dict = {}
            member_dict = {}
            agent_dict_from_data = {}
            member_dict_from_data = {}

            event_content = event_content.replace("pogs3.2:", "pogs3.2=")

            if "=+" in event_content:
                event_content = event_content.replace("=+", "=25(filled)+")
            if '="' in event_content:
                event_content = event_content.replace('="', '=25(filled)"')
            if '= ' in event_content:
                event_content = event_content.replace('= ', '=0 (filled)')
            if 'undefined' in event_content:
                event_content = event_content.replace('undefined', '0(filled)')

            event_content_array = event_content.replace("{", "").replace("}", "").split(" ")
            agent_line = event_content_array[2]
            member_line = event_content_array[5].split("||")[0].replace('"', "")

            members = []
            for k, v in (item.split("=") for item in agent_line.split("+")):
                members.append(k)

            # Fix the case where we didn't see all members earlier, so add to the mapping
            for k in members:
                if k not in self.team_member_mapping.keys():
                    self._add_to_team_member_mapping(k)


            for str in agent_line.split("+"):
                val = True
                if ("(filled)" in str):
                    val = False
                str_array = str.split("=")
                agent_dict_from_data[self.team_member_mapping[str_array[0]]] = val

            for str in member_line.split("+"):
                val = True
                if ("(filled)" in str):
                    val = False
                str_array = str.split("=")
                member_dict_from_data[self.team_member_mapping[str_array[0]]] = val

            event_content = event_content.replace("(filled)", "")
            event_content_array = event_content.replace("{", "").replace("}", "").split(" ")
            agent_line = event_content_array[2]
            member_line = event_content_array[5].split("||")[0].replace('"', "")

            personal_agent_ratings_dict = dict((self.team_member_mapping[k], float(v)) for k, v in (item.split("=") for item in agent_line.split("+")))
            personal_member_ratings_dict = dict((self.team_member_mapping[k], float(v)) for k, v in (item.split("=") for item in member_line.split("+")))
            agent_dict[team_member_id] = personal_agent_ratings_dict
            member_dict[team_member_id] = personal_member_ratings_dict

            return agent_dict, member_dict, agent_dict_from_data, member_dict_from_data, team_member_id

        self.agent_ratings = {}
        self.member_ratings = {}
        self.agent_ratings_from_data = {}
        self.member_ratings_from_data = {}
        ratings_index = 0
        while end_index < len(indices):
            if indices[end_index] - indices[begin_index] > 4:
                df = self.team_event_logs.iloc[
                    indices[begin_index] + 1: indices[end_index]]
                df = df[df.extra_data == 'InfluenceMatrix']

                if (not df.empty):
                    if (len(df.index) >=4):
                        try:
                            self.team_member_mapping
                        except AttributeError:
                            self._create_team_member_mapping(df)
                        agent_dict_list = []
                        member_dict_list = []
                        agent_dict_from_data_list = []
                        member_dict_from_data_list = []
                        team_member_ids = []
                        for index in range(len(df)):
                            agent_dict, member_dict, agent_dict_from_data, member_dict_from_data, team_member_id = \
                                parse_event_content(str(df.event_content.iloc[index]), int(str(df.sender_subject_id.iloc[index])))

                            team_member_ids.append(team_member_id)

                            # Handle duplicates
                            if agent_dict not in agent_dict_list:
                                agent_dict_list.append(agent_dict)
                            if member_dict not in member_dict_list:
                                member_dict_list.append(member_dict)
                            if agent_dict_from_data not in agent_dict_from_data_list:
                                agent_dict_from_data_list.append(agent_dict)
                            if member_dict_from_data not in agent_dict_from_data_list:
                                agent_dict_from_data_list.append(member_dict)

                        print("agent_dict_list = ", agent_dict_list)

                        # If we're missing an entry, add one with default values of 0 for agent and 25 for members and record that we filled in
                        for k in self.team_member_mapping.keys():
                            v = self.team_member_mapping[k]
                            if v not in team_member_ids:
                                #agent_dict_list[]

                                fake_agent_values = {}
                                for id in self.team_member_mapping.values():
                                    fake_agent_values[id] = 0
                                personal_fake_agent_values = {}
                                personal_fake_agent_values[v] = fake_agent_values
                                agent_dict_list.append(personal_fake_agent_values)

                                agent_dict_from_data[v] = False

                                fake_member_values = {}
                                for id in self.team_member_mapping.values():
                                    fake_member_values[id] = 25
                                personal_fake_member_values = {}
                                personal_fake_member_values[v] = fake_member_values
                                member_dict_list.append(personal_fake_member_values)

                                member_dict_from_data[v] = False

                        self.agent_ratings[ratings_index] = agent_dict_list
                        self.member_ratings[ratings_index] = member_dict_list
                        self.agent_ratings_from_data[ratings_index] = agent_dict_from_data_list
                        self.member_ratings_from_data[ratings_index] = member_dict_from_data_list

                        ratings_index = ratings_index + 1
                    else:
                        print("invalid for index = " + str(begin_index))
                        return

            begin_index = end_index
            end_index += 1

    def _load_accumulated_score(self) -> None:
        """Loads the accumulated score per question"""
        indices = [0] + list(
            np.where(self.team_event_logs.extra_data == 'SubmitButtonField')[0])
        begin_index = 0
        end_index = 1

        # def parse_event_content(event_content):

        self.score = {}
        self.accumulated_score = {}
        self.accumulated_score[0] = 0
        index = 1
        for i in self.question_order:

            score_earned = 0
            final_answer_chosen = None
            if len(set(self.group_answers_chosen[i].values())) == 1:
                final_answer_chosen = self.group_answers_chosen[i][list(self.group_answers_chosen[i].keys())[0]]

            answer = None
            for questions_index in range(len(self.game_info.questions)):
                if (i == self.game_info.questions[questions_index].id):
                     answer = self.game_info.questions[questions_index].answer

            if (final_answer_chosen == answer):
                score_earned = score_earned + 4
            else:
                score_earned = score_earned - 1

            if self.machine_usage_info[i].used:
                score_earned = score_earned - 1

            self.score[i] = score_earned
            self.accumulated_score[index] = self.accumulated_score[index - 1] + score_earned
            index = index + 1


    def _load_influence_matrices(self) -> None:
        pass
#         team_size = self.game_info.num_of_team_members   # For convenience.
#         self.agent_ratings = list()
#         self.member_influences = list()
#         mInfluences = [0 for i in range(team_size)]
#         aRatings = [0 for i in range(team_size)]
#         count = 0
#         influenceMatrices = self.team_event_logs[(self.team_event_logs['extra_data'] == "InfluenceMatrix")]  
#         influenceMatrixWithoutUndefined = influenceMatrices[~influenceMatrices['stringValue'].str.contains("undefined")]
#         finalInfluences = influenceMatrixWithoutUndefined.groupby(['questionScore', 'sender'], as_index=False, sort=False).last()
#         for i in range(len(finalInfluences)):
#             count +=1 
#             aR = list()
#             mI = list() 
#             idx = self.teamArray.index(finalInfluences.iloc[i]['sender'])
#             for j in range(0, team_size):
#                 temp = finalInfluences.iloc[i]['stringValue']
# #                 Fill missing values
#                 xy = re.findall(r'Ratings(.*?) Member', temp)[0].split("+")[j].split("=")[1]
#                 if(xy==''):
#                     xy = '0.5'
#                 yz= temp.replace('"', '')[temp.index("Influences ")+10:].split("+")[j].split("=")[1]
#                 if(yz == ''):
#                     yz = '25'
#                 aR.append(float(xy))
#                 mI.append(int(round(float(yz))))
#             aRatings[idx]=aR
#             mInfluences[idx]=mI 
#             if(count % team_size == 0):
#                 self.member_influences.append(mInfluences)
#                 mInfluences = [0 for i in range(team_size)]
#                 self.agent_ratings.append(aRatings)
#                 aRatings = [0 for i in range(team_size)]

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
        eventLog = pd.read_csv(directory+"event_log.csv", sep=',',quotechar="|", names=["id","event_type","event_content","timestamp","completed_task_id","sender_subject_id","receiver_subject_id","session_id","sender","receiver","extra_data"])
        teamSubjects = pd.read_csv(directory+"team_has_subject.csv",sep=',',quotechar="|",names=["id","teamId","sender_subject_id"]).drop('id',1)
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
        with open(directory+"jeopardy.json") as json_data:
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
        subjects = pd.read_csv(directory+"subject.csv", sep=',',quotechar="|", names=["sender_subject_id","externalId","displayName","sessionId","previousSessionSubject"])
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
