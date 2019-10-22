# Omid55
# Test module for pogs jeopardy log library.
from __future__ import division, print_function, absolute_import, unicode_literals

import pandas as pd
import numpy as np
import mock
import os
import unittest
from parameterized import parameterized

import pogs_jeopardy_log_lib as lib


TESTING_LOG = '''2298,COMMUNICATION_MESSAGE,{"channel":null||"message":""||"type":"JOINED"},2019-05-02 10:12:09,32,20,\\N,4,pogs3.4,\\N,\\N
    2299,COMMUNICATION_MESSAGE,{"channel":null||"message":""||"type":"JOINED"},2019-05-02 10:12:09,32,18,\\N,4,pogs3.2,\\N,\\N
    2300,COMMUNICATION_MESSAGE,{"channel":null||"message":""||"type":"JOINED"},2019-05-02 10:12:09,32,17,\\N,4,pogs3.1,\\N,\\N
    2301,COMMUNICATION_MESSAGE,{"channel":null||"message":""||"type":"JOINED"},2019-05-02 10:12:09,32,19,\\N,4,pogs3.3,\\N,\\N
    2302,TASK_ATTRIBUTE,{"attributeStringValue":"Radio Waves"||"attributeDoubleValue":1.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":0||"loggableAttribute":true},2019-05-02 10:12:15,32,17,\\N,4,pogs3.1,\\N,IndividualResponse
    2303,TASK_ATTRIBUTE,{"attributeStringValue":"Radio Waves"||"attributeDoubleValue":1.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":0||"loggableAttribute":true},2019-05-02 10:12:16,32,20,\\N,4,pogs3.4,\\N,IndividualResponse
    2304,TASK_ATTRIBUTE,{"attributeStringValue":"Radio Waves"||"attributeDoubleValue":1.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":0||"loggableAttribute":true},2019-05-02 10:12:18,32,19,\\N,4,pogs3.3,\\N,IndividualResponse
    2305,TASK_ATTRIBUTE,{"attributeStringValue":"Radio Waves"||"attributeDoubleValue":1.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":0||"loggableAttribute":true},2019-05-02 10:12:31,32,18,\\N,4,pogs3.2,\\N,IndividualResponse
    2306,TASK_ATTRIBUTE,{"attributeStringValue":"Radio Waves"||"attributeDoubleValue":1.0||"attributeName":"jeopardyAnswer0__pogs3.4"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:13:28,32,20,\\N,4,pogs3.4,\\N,SubmitButtonField
    2307,TASK_ATTRIBUTE,{"attributeStringValue":"The Merchant of Venice"||"attributeDoubleValue":43.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:13:42,32,20,\\N,4,pogs3.4,\\N,IndividualResponse
    2308,TASK_ATTRIBUTE,{"attributeStringValue":"The Tempest"||"attributeDoubleValue":43.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:13:43,32,17,\\N,4,pogs3.1,\\N,IndividualResponse
    2309,TASK_ATTRIBUTE,{"attributeStringValue":"The Merchant of Venice"||"attributeDoubleValue":43.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:13:46,32,18,\\N,4,pogs3.2,\\N,IndividualResponse
    2310,TASK_ATTRIBUTE,{"attributeStringValue":"The Tempest"||"attributeDoubleValue":43.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:13:47,32,19,\\N,4,pogs3.3,\\N,IndividualResponse
    2311,TASK_ATTRIBUTE,{"attributeStringValue":"The Merchant of Venice"||"attributeDoubleValue":43.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:13:49,32,17,\\N,4,pogs3.1,\\N,IndividualResponse
    2312,TASK_ATTRIBUTE,{"attributeStringValue":"The Tempest"||"attributeDoubleValue":43.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:13:50,32,17,\\N,4,pogs3.1,\\N,IndividualResponse
    2313,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"What do you think?"||"type":"MESSAGE"},2019-05-02 10:14:13,32,20,\\N,4,pogs3.4,\\N,\\N
    2314,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"i was 50\\/50 between the two"||"type":"MESSAGE"},2019-05-02 10:14:25,32,17,\\N,4,pogs3.1,\\N,\\N
    2315,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"i guessed so i'm not sure"||"type":"MESSAGE"},2019-05-02 10:14:27,32,18,\\N,4,pogs3.2,\\N,\\N
    2316,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"I have no idea I just know its not hamlet"||"type":"MESSAGE"},2019-05-02 10:14:29,32,19,\\N,4,pogs3.3,\\N,\\N
    2317,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"I put The Merchant of Venice because the names sound Italian"||"type":"MESSAGE"},2019-05-02 10:14:33,32,20,\\N,4,pogs3.4,\\N,\\N
    2318,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"I was 50\\/50 too"||"type":"MESSAGE"},2019-05-02 10:14:42,32,19,\\N,4,pogs3.3,\\N,\\N
    2319,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"should we ask the machine?"||"type":"MESSAGE"},2019-05-02 10:14:55,32,18,\\N,4,pogs3.2,\\N,\\N
    2320,TASK_ATTRIBUTE,{"attributeStringValue":"The Merchant of Venice"||"attributeDoubleValue":43.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:14:56,32,19,\\N,4,pogs3.3,\\N,GroupRadioResponse
    2321,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"its also 50\\/50 tho"||"type":"MESSAGE"},2019-05-02 10:15:12,32,19,\\N,4,pogs3.3,\\N,\\N
    2322,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"I think we should just submit"||"type":"MESSAGE"},2019-05-02 10:15:21,32,20,\\N,4,pogs3.4,\\N,\\N
    2323,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"im pretty sure its tempest"||"type":"MESSAGE"},2019-05-02 10:15:22,32,17,\\N,4,pogs3.1,\\N,\\N
    2324,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"ok"||"type":"MESSAGE"},2019-05-02 10:15:26,32,18,\\N,4,pogs3.2,\\N,\\N
    2325,TASK_ATTRIBUTE,{"attributeStringValue":"The Merchant of Venice"||"attributeDoubleValue":43.0||"attributeName":"jeopardyAnswer0__pogs3.4"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:15:27,32,20,\\N,4,pogs3.4,\\N,SubmitButtonField
    2326,TASK_ATTRIBUTE,{"attributeStringValue":"The Merchant of Venice"||"attributeDoubleValue":43.0||"attributeName":"jeopardyAnswer0__pogs3.2"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:15:27,32,18,\\N,4,pogs3.2,\\N,SubmitButtonField
    2327,TASK_ATTRIBUTE,{"attributeStringValue":"The Merchant of Venice"||"attributeDoubleValue":43.0||"attributeName":"jeopardyAnswer0__pogs3.3"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:15:27,32,19,\\N,4,pogs3.3,\\N,SubmitButtonField
    2328,TASK_ATTRIBUTE,{"attributeStringValue":"Roosevelt"||"attributeDoubleValue":26.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:15:33,32,17,\\N,4,pogs3.1,\\N,IndividualResponse
    2329,TASK_ATTRIBUTE,{"attributeStringValue":"Roosevelt"||"attributeDoubleValue":26.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:15:47,32,18,\\N,4,pogs3.2,\\N,IndividualResponse
    2330,TASK_ATTRIBUTE,{"attributeStringValue":"Roosevelt"||"attributeDoubleValue":26.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:15:47,32,19,\\N,4,pogs3.3,\\N,IndividualResponse
    2331,TASK_ATTRIBUTE,{"attributeStringValue":"Roosevelt"||"attributeDoubleValue":26.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:15:49,32,20,\\N,4,pogs3.4,\\N,IndividualResponse
    2332,TASK_ATTRIBUTE,{"attributeStringValue":"Roosevelt"||"attributeDoubleValue":26.0||"attributeName":"jeopardyAnswer0__pogs3.3"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:16:47,32,19,\\N,4,pogs3.3,\\N,SubmitButtonField
    2333,TASK_ATTRIBUTE,{"attributeStringValue":"Chicken Little"||"attributeDoubleValue":41.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:16:54,32,20,\\N,4,pogs3.4,\\N,IndividualResponse
    2334,TASK_ATTRIBUTE,{"attributeStringValue":"Chicken Little"||"attributeDoubleValue":41.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:16:59,32,19,\\N,4,pogs3.3,\\N,IndividualResponse
    2335,TASK_ATTRIBUTE,{"attributeStringValue":"Chicken Little"||"attributeDoubleValue":41.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:17:07,32,18,\\N,4,pogs3.2,\\N,IndividualResponse
    2336,TASK_ATTRIBUTE,{"attributeStringValue":"Chicken Little"||"attributeDoubleValue":41.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:17:20,32,17,\\N,4,pogs3.1,\\N,IndividualResponse
    2337,TASK_ATTRIBUTE,{"attributeStringValue":"A Goofy Movies"||"attributeDoubleValue":41.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:17:23,32,18,\\N,4,pogs3.2,\\N,IndividualResponse
    2338,TASK_ATTRIBUTE,{"attributeStringValue":"Chicken Little"||"attributeDoubleValue":41.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:17:26,32,18,\\N,4,pogs3.2,\\N,IndividualResponse
    2339,TASK_ATTRIBUTE,{"attributeStringValue":"Chicken Little"||"attributeDoubleValue":41.0||"attributeName":"jeopardyAnswer0__pogs3.3"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:18:25,32,19,\\N,4,pogs3.3,\\N,SubmitButtonField
    2340,TASK_ATTRIBUTE,{"attributeStringValue":"Helium"||"attributeDoubleValue":4.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:18:29,32,20,\\N,4,pogs3.4,\\N,IndividualResponse
    2341,TASK_ATTRIBUTE,{"attributeStringValue":"Helium"||"attributeDoubleValue":4.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:18:30,32,18,\\N,4,pogs3.2,\\N,IndividualResponse
    2342,TASK_ATTRIBUTE,{"attributeStringValue":"oxygen"||"attributeDoubleValue":4.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:18:36,32,19,\\N,4,pogs3.3,\\N,IndividualResponse
    2343,TASK_ATTRIBUTE,{"attributeStringValue":"Helium"||"attributeDoubleValue":4.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:18:42,32,19,\\N,4,pogs3.3,\\N,IndividualResponse
    2344,TASK_ATTRIBUTE,{"attributeStringValue":"Helium"||"attributeDoubleValue":4.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:18:45,32,17,\\N,4,pogs3.1,\\N,IndividualResponse
    2345,TASK_ATTRIBUTE,{"attributeStringValue":"Helium"||"attributeDoubleValue":4.0||"attributeName":"jeopardyAnswer0__pogs3.2"||"attributeIntegerValue":10||"loggableAttribute":true},2019-05-02 10:19:41,32,18,\\N,4,pogs3.2,\\N,SubmitButtonField
    2346,TASK_ATTRIBUTE,{"attributeStringValue":"Agent Ratings pogs3.2=0.3+pogs3.4=0.1+pogs3.3=0.3+pogs3.1=0.3 Member Influences pogs3.2=40+pogs3.4=0+pogs3.3=40+pogs3.1=20"||"attributeDoubleValue":0.0||"attributeName":"jeopardyAnswer0__pogs3.4"||"attributeIntegerValue":1||"loggableAttribute":true},2019-05-02 10:23:06,32,20,\\N,4,pogs3.4,\\N,InfluenceMatrix
    2347,TASK_ATTRIBUTE,{"attributeStringValue":"Agent Ratings pogs3.2=0+pogs3.4=0+pogs3.3=0+pogs3.1=0 Member Influences pogs3.2=1+pogs3.4=1+pogs3.3=1+pogs3.1=97"||"attributeDoubleValue":0.0||"attributeName":"jeopardyAnswer0__pogs3.1"||"attributeIntegerValue":1||"loggableAttribute":true},2019-05-02 10:23:08,32,17,\\N,4,pogs3.1,\\N,InfluenceMatrix
    2348,TASK_ATTRIBUTE,{"attributeStringValue":"Agent Ratings pogs3.2=0+pogs3.4=0+pogs3.3=0+pogs3.1=0 Member Influences pogs3.2=1+pogs3.4=1+pogs3.3=1+pogs3.1=97"||"attributeDoubleValue":0.0||"attributeName":"jeopardyAnswer0__pogs3.1"||"attributeIntegerValue":1||"loggableAttribute":true},2019-05-02 10:23:09,32,17,\\N,4,pogs3.1,\\N,InfluenceMatrix
    2349,TASK_ATTRIBUTE,{"attributeStringValue":"Agent Ratings pogs3.2=0+pogs3.4=0+pogs3.3=0+pogs3.1=0 Member Influences pogs3.2=0+pogs3.4=100+pogs3.3=0+pogs3.1=0"||"attributeDoubleValue":0.0||"attributeName":"jeopardyAnswer0__pogs3.3"||"attributeIntegerValue":1||"loggableAttribute":true},2019-05-02 10:23:09,32,19,\\N,4,pogs3.3,\\N,InfluenceMatrix
    2350,TASK_ATTRIBUTE,{"attributeStringValue":"Best Sound Mixing"||"attributeDoubleValue":42.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:23:17,32,20,\\N,4,pogs3.4,\\N,IndividualResponse
    2351,TASK_ATTRIBUTE,{"attributeStringValue":"Best Sound Mixing"||"attributeDoubleValue":42.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:23:18,32,19,\\N,4,pogs3.3,\\N,IndividualResponse
    2352,TASK_ATTRIBUTE,{"attributeStringValue":"Best Sound Mixing"||"attributeDoubleValue":42.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:23:21,32,17,\\N,4,pogs3.1,\\N,IndividualResponse
    2353,TASK_ATTRIBUTE,{"attributeStringValue":"Best Original Score"||"attributeDoubleValue":42.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:23:24,32,18,\\N,4,pogs3.2,\\N,IndividualResponse
    2354,TASK_ATTRIBUTE,{"attributeStringValue":"Best Sound Mixing"||"attributeDoubleValue":42.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":5||"loggableAttribute":true},2019-05-02 10:23:46,32,18,\\N,4,pogs3.2,\\N,GroupRadioResponse
    2355,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"I guessed on this one "||"type":"MESSAGE"},2019-05-02 10:23:52,32,20,\\N,4,pogs3.4,\\N,\\N
    2356,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"Should I ask the machine?"||"type":"MESSAGE"},2019-05-02 10:24:00,32,20,\\N,4,pogs3.4,\\N,\\N
    2357,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"i feel like ive heard original score being a catergory"||"type":"MESSAGE"},2019-05-02 10:24:09,32,19,\\N,4,pogs3.3,\\N,\\N
    2358,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"We have less than 5 minutes on this round I think"||"type":"MESSAGE"},2019-05-02 10:24:11,32,20,\\N,4,pogs3.4,\\N,\\N
    2359,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"all the other ones still exist"||"type":"MESSAGE"},2019-05-02 10:24:15,32,17,\\N,4,pogs3.1,\\N,\\N
    2360,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"i guessed too. I don't watch the oscars."||"type":"MESSAGE"},2019-05-02 10:24:18,32,18,\\N,4,pogs3.2,\\N,\\N
    2361,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"go ahead and ask it"||"type":"MESSAGE"},2019-05-02 10:24:22,32,19,\\N,4,pogs3.3,\\N,\\N
    2362,TASK_ATTRIBUTE,{"attributeStringValue":"AskMachine:Best Sound Mixing_0.6"||"attributeDoubleValue":42.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":4||"loggableAttribute":true},2019-05-02 10:24:25,32,20,\\N,4,pogs3.4,\\N,AskedMachine
    2363,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"it says best sound mixing"||"type":"MESSAGE"},2019-05-02 10:24:32,32,20,\\N,4,pogs3.4,\\N,\\N
    2364,TASK_ATTRIBUTE,{"attributeStringValue":"Best Sound Mixing"||"attributeDoubleValue":42.0||"attributeName":"jeopardyAnswer0__pogs3.1"||"attributeIntegerValue":4||"loggableAttribute":true},2019-05-02 10:24:39,32,17,\\N,4,pogs3.1,\\N,SubmitButtonField
    2365,TASK_ATTRIBUTE,{"attributeStringValue":"DIFFERENT TEAM MEMBER MISTAKABLY ADDED"||"attributeDoubleValue":42.0||"attributeName":"jeopardyAnswer0__pogs3999.1"||"attributeIntegerValue":4||"loggableAttribute":true},2019-05-02 10:24:39,32,999,\\N,999,pogs3999.1,\\N,SubmitButtonField
    2366,TASK_ATTRIBUTE,{"attributeStringValue":"DIFFERENT TEAM MEMBER MISTAKABLY ADDED 2"||"attributeDoubleValue":42.0||"attributeName":"jeopardyAnswer0__pogs3999.1"||"attributeIntegerValue":4||"loggableAttribute":true},2019-05-02 10:24:39,32,999,\\N,999,pogs3999.1,\\N,SubmitButtonField'''
TESTING_JEOPARDY_JSON = '''
    [
        {
            "ID":1
            ,"question":"Which kind of waves are used to make and receive cellphone calls?"
            ,"Answer":"Radio Waves"
            ,"value":["Radio Waves", "Sound Waves", "Gravity Waves", "Visible Light Waves"]
            ,"Category":"Science and Technology"
            ,"Level":"Easy"
        },
        {
            "ID":4
            ,"question":"Balloons are filled with"
            ,"Answer":"Helium"
            ,"value":["oxygen", "nitrogen", "argon", "Helium"]
            ,"Category":"Science and Technology"
            ,"Level":"Easy"
        },
        {
            "ID":26
            ,"question":"Before Eisenhower, he was the last president to preside over the admission of a new state."
            ,"Answer":"Taft"
            ,"value":["Roosevelt", "Lincoln", "Eisenhower", "Taft"]
            ,"Category":"History and Mythology"
            ,"Level":"Hard"
        },
        {
            "ID":41
            ,"question":"This Disney movie was not created by Walt Disney Animation Studios."
            ,"Answer":"A Goofy Movies"
            ,"value":["Frozen", "Hercules", "Chicken Little", "A Goofy Movies"]
            ,"Category":"Literature and Media"
            ,"Level":"Hard"
        },
        {
            "ID":42
            ,"question":"This Oscar category has been discontinued"
            ,"Answer":"Best Assistant Director"
            ,"value":["Best Original Score", "Best Cinematography", "Best Sound Mixing", "Best Assistant Director"]
            ,"Category":"Literature and Media"
            ,"Level":"Hard"
        },
        {
            "ID":43
            ,"question":"In which Shakespeare play are Stephano and Trinculo characters?"
            ,"Answer":"The Tempest"
            ,"value":["The Tempest", "Hamlet", "As You Like It", "The Merchant of Venice"]
            ,"Category":"Literature and Media"
            ,"Level":"Hard"
        }
    ]
    '''
TESTING_TEAM_HAS_SUBJECT = '''
    25,1,18
    26,1,20
    27,1,19
    28,1,17
    '''
TESTING_LOG_FILE_PATH = '/tmp/event_log.csv'
TESTING_JEOPARDY_FILE_PATH = '/tmp/jeopardy.json'
TESTING_TEAM_HAS_SUBJECT_FILE_PATH = '/tmp/team_has_subject.csv'


# =========================================================================
# =========================== _load_game_questions ========================
# =========================================================================
class TeamLogProcessorLoadGameQuestionsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):        
        # with open(TESTING_LOG_FILE_PATH, 'w') as f:
        #     f.writelines(TESTING_LOG)
        with open(TESTING_JEOPARDY_FILE_PATH, 'w') as f:
            f.writelines(TESTING_JEOPARDY_JSON)
        with mock.patch.object(lib.TeamLogProcessor, '_load_all_files'):
            cls.loader = lib.TeamLogProcessor(
                team_id=1, logs_directory_path='tmp')
            cls.loader._load_game_questions(TESTING_JEOPARDY_FILE_PATH)

    @classmethod
    def tearDownClass(cls):
        # os.remove(TESTING_LOG_FILE_PATH)
        os.remove(TESTING_JEOPARDY_FILE_PATH)

    def test_load_game_questions_has_correctly_updated_num_of_questions(self):
        self.assertEqual(self.loader.game_info.num_of_questions, 6)

    def test_load_game_questions_has_correct_number_of_questions(self):
        self.assertEqual(len(self.loader.game_info.questions), 6)

    def test_load_game_questions_has_correct_id(self):
        self.assertEqual(self.loader.game_info.questions[2].id, 26)

    def test_load_game_questions_has_correct_content(self):
        self.assertEqual(
            self.loader.game_info.questions[3].question_content,
            'This Disney movie was not created by Walt Disney Animation'
            ' Studios.')

    def test_load_game_questions_has_correct_answer(self):
        self.assertEqual(
            self.loader.game_info.questions[0].answer, 'Radio Waves')

    def test_load_game_questions_has_correct_choices(self):
        self.assertEqual(
            self.loader.game_info.questions[1].choices,
            ['oxygen', 'nitrogen', 'argon', 'Helium'])

    def test_load_game_questions_has_correct_category(self):
        self.assertEqual(
            self.loader.game_info.questions[4].category,
            lib.Category.LITERATURE_AND_MEDIA)
    
    def test_load_game_questions_has_correct_level(self):
        self.assertEqual(
            self.loader.game_info.questions[5].level,
            lib.Level.HARD)


# =========================================================================
# ======================= _load_this_team_event_logs ======================
# =========================================================================
class TeamLogProcessorLoadThisTeamEventLogTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):        
        with open(TESTING_LOG_FILE_PATH, 'w') as f:
            f.writelines(TESTING_LOG)
        with open(TESTING_TEAM_HAS_SUBJECT_FILE_PATH, 'w') as f:
            f.writelines(TESTING_TEAM_HAS_SUBJECT)
        with mock.patch.object(lib.TeamLogProcessor, '_load_all_files'):
            cls.loader = lib.TeamLogProcessor(
                team_id=1, logs_directory_path='tmp')

    @classmethod
    def tearDownClass(cls):
        os.remove(TESTING_LOG_FILE_PATH)
        os.remove(TESTING_TEAM_HAS_SUBJECT_FILE_PATH)

    def test_load_this_team_event_logs_loads_team_logs_correctly(self):
        self.loader._load_this_team_event_logs(
            logs_file_path=TESTING_LOG_FILE_PATH,
            team_has_subject_file_path=TESTING_TEAM_HAS_SUBJECT_FILE_PATH)
        lines_num = len(TESTING_LOG.split('\n'))
        self.assertEqual(len(self.loader.team_event_logs), lines_num - 2)


# =========================================================================
# ======================== _load_messages =================================
# =========================================================================
class TeamLogProcessorLoadMessagesTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):        
        with open(TESTING_LOG_FILE_PATH, 'w') as f:
            f.writelines(TESTING_LOG)
        with open(TESTING_TEAM_HAS_SUBJECT_FILE_PATH, 'w') as f:
            f.writelines(TESTING_TEAM_HAS_SUBJECT)
        with mock.patch.object(lib.TeamLogProcessor, '_load_all_files'):
            cls.loader = lib.TeamLogProcessor(
                team_id=1, logs_directory_path='tmp')
            cls.loader._load_this_team_event_logs(
                logs_file_path=TESTING_LOG_FILE_PATH,
                team_has_subject_file_path=TESTING_TEAM_HAS_SUBJECT_FILE_PATH)
            cls.loader._load_messages()

    @classmethod
    def tearDownClass(cls):
        os.remove(TESTING_LOG_FILE_PATH)
        os.remove(TESTING_TEAM_HAS_SUBJECT_FILE_PATH)

    def test_load_messages_has_loaded_log_correctly(self):
        messages = self.loader.messages  # Just for the convenice.
        self.assertEqual(len(messages), 6)
        self.assertEqual(messages[0].shape, (0, 12))
        self.assertEqual(messages[1].shape, (11, 12))
        self.assertEqual(messages[2].shape, (0, 12))
        self.assertEqual(messages[3].shape, (0, 12))
        self.assertEqual(messages[4].shape, (0, 12))
        self.assertEqual(messages[5].shape, (8, 12))

# =========================================================================
# ======================== _load_answers_chosen ===========================
# =========================================================================
class TeamLogProcessorLoadAnswersChosenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):        
        with open(TESTING_LOG_FILE_PATH, 'w') as f:
            f.writelines(TESTING_LOG)
        with open(TESTING_TEAM_HAS_SUBJECT_FILE_PATH, 'w') as f:
            f.writelines(TESTING_TEAM_HAS_SUBJECT)
        with open(TESTING_JEOPARDY_FILE_PATH, 'w') as f:
            f.writelines(TESTING_JEOPARDY_JSON)
        with mock.patch.object(lib.TeamLogProcessor, '_load_all_files'):
            cls.loader = lib.TeamLogProcessor(
                team_id=1, logs_directory_path='tmp')
            cls.loader._load_this_team_event_logs(
                logs_file_path=TESTING_LOG_FILE_PATH,
                team_has_subject_file_path=TESTING_TEAM_HAS_SUBJECT_FILE_PATH)
            cls.loader._load_answers_chosen()

    @classmethod
    def tearDownClass(cls):
        os.remove(TESTING_LOG_FILE_PATH)
        os.remove(TESTING_TEAM_HAS_SUBJECT_FILE_PATH)
        os.remove(TESTING_JEOPARDY_FILE_PATH)

    def test_load_answers_chosen_has_loaded_correctly(self):
        individual_responses = self.loader.individual_answers_chosen
        group_responses = self.loader.group_answers_chosen
        self.assertEqual(len(individual_responses), 6)
        self.assertEqual(len(group_responses), 6)

        keys = [1, 4, 26, 41, 42, 43]
        self.assertEqual(all(key in keys for key in individual_responses.keys()), True)
        self.assertEqual(all(key in individual_responses.keys() for key in keys), True)
        self.assertEqual(all(key in keys for key in group_responses.keys()), True)
        self.assertEqual(all(key in group_responses.keys() for key in keys), True)
        self.assertEqual(individual_responses[41] == {17: 'Chicken Little', 18: 'Chicken Little', 19: 'Chicken Little', 20: 'Chicken Little'}, True)
        self.assertEqual(group_responses[41] == {20: 'Chicken Little', 19: 'Chicken Little', 18: 'Chicken Little', 17: 'Chicken Little'}, True)
        self.assertEqual(individual_responses[42] == {17: 'Best Sound Mixing', 18: 'Best Original Score', 19: 'Best Sound Mixing', 20: 'Best Sound Mixing'}, True)
        self.assertEqual(group_responses[42] == {19: 'Best Sound Mixing', 17: 'Best Sound Mixing', 20: 'Best Sound Mixing', 18: 'Best Sound Mixing'}, True)

# # =========================================================================
# # ==================== _load_influence_matrices ===========================
# # =========================================================================
# class TeamLogProcessorLoadInfluenceMatrixTest(unittest.TestCase):

#     @classmethod
#     def setUpClass(cls):        
#         with open(TESTING_LOG_FILE_PATH, 'w') as f:
#             f.writelines(TESTING_LOG)
#         with open(TESTING_TEAM_HAS_SUBJECT_FILE_PATH, 'w') as f:
#             f.writelines(TESTING_TEAM_HAS_SUBJECT)
#         with mock.patch.object(lib.TeamLogProcessor, '_load_all_files'):
#             cls.loader = lib.TeamLogProcessor(
#                 team_id=1, logs_directory_path='tmp')
#             cls.loader._load_this_team_event_logs(
#                 logs_file_path=TESTING_LOG_FILE_PATH,
#                 team_has_subject_file_path=TESTING_TEAM_HAS_SUBJECT_FILE_PATH)
#             cls.loader._load_influence_matrices()

#     @classmethod
#     def tearDownClass(cls):
#         os.remove(TESTING_LOG_FILE_PATH)
#         os.remove(TESTING_TEAM_HAS_SUBJECT_FILE_PATH)

#     def test_load_influence_matrices_has_loaded_log_correctly(self):
#         # self.loader.member_influences
#         pass
