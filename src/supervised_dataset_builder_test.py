from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import pandas as pd
import pandas.testing as pd_testing
import unittest
import os
from os.path import expanduser
from parameterized import parameterized

import text_processor
import supervised_dataset_builder as data_builder
import pogs_jeopardy_log_lib as pogs_lib

TESTING_LOG = '''
    2279,TASK_ATTRIBUTE,{"attributeStringValue":"3 - Somewhat"||"attributeDoubleValue":0.0||"attributeName":"surveyAnswer0"||"attributeIntegerValue":-1||"loggableAttribute":true},2019-05-02 10:11:23,30,20,\\N,4,pogs3.4,\\N,RadioField
    2280,TASK_ATTRIBUTE,{"attributeStringValue":"2 - Slightly"||"attributeDoubleValue":0.0||"attributeName":"surveyAnswer1"||"attributeIntegerValue":-1||"loggableAttribute":true},2019-05-02 10:11:24,30,20,\\N,4,pogs3.4,\\N,RadioField
    2281,TASK_ATTRIBUTE,{"attributeStringValue":"2 - Slightly"||"attributeDoubleValue":0.0||"attributeName":"surveyAnswer0"||"attributeIntegerValue":-1||"loggableAttribute":true},2019-05-02 10:11:24,29,19,\\N,4,pogs3.3,\\N,RadioField
    2282,TASK_ATTRIBUTE,{"attributeStringValue":"1- Not at all"||"attributeDoubleValue":0.0||"attributeName":"surveyAnswer1"||"attributeIntegerValue":-1||"loggableAttribute":true},2019-05-02 10:11:25,30,20,\\N,4,pogs3.4,\\N,RadioField
    2283,TASK_ATTRIBUTE,{"attributeStringValue":"3 - Somewhat"||"attributeDoubleValue":0.0||"attributeName":"surveyAnswer0"||"attributeIntegerValue":-1||"loggableAttribute":true},2019-05-02 10:11:25,28,18,\\N,4,pogs3.2,\\N,RadioField
    2284,TASK_ATTRIBUTE,{"attributeStringValue":"2 - Slightly"||"attributeDoubleValue":0.0||"attributeName":"surveyAnswer2"||"attributeIntegerValue":-1||"loggableAttribute":true},2019-05-02 10:11:27,30,20,\\N,4,pogs3.4,\\N,RadioField
    2285,TASK_ATTRIBUTE,{"attributeStringValue":"3 - Somewhat"||"attributeDoubleValue":0.0||"attributeName":"surveyAnswer1"||"attributeIntegerValue":-1||"loggableAttribute":true},2019-05-02 10:11:28,28,18,\\N,4,pogs3.2,\\N,RadioField
    2286,TASK_ATTRIBUTE,{"attributeStringValue":"4 - Fairly Well"||"attributeDoubleValue":0.0||"attributeName":"surveyAnswer1"||"attributeIntegerValue":-1||"loggableAttribute":true},2019-05-02 10:11:30,29,19,\\N,4,pogs3.3,\\N,RadioField
    2287,TASK_ATTRIBUTE,{"attributeStringValue":"1- Not at all"||"attributeDoubleValue":0.0||"attributeName":"surveyAnswer2"||"attributeIntegerValue":-1||"loggableAttribute":true},2019-05-02 10:11:31,28,18,\\N,4,pogs3.2,\\N,RadioField
    2288,TASK_ATTRIBUTE,{"attributeStringValue":"3 - Somewhat"||"attributeDoubleValue":0.0||"attributeName":"surveyAnswer2"||"attributeIntegerValue":-1||"loggableAttribute":true},2019-05-02 10:11:32,29,19,\\N,4,pogs3.3,\\N,RadioField
    2289,TASK_ATTRIBUTE,{"attributeStringValue":"4 - Fairly Well"||"attributeDoubleValue":0.0||"attributeName":"surveyAnswer2"||"attributeIntegerValue":-1||"loggableAttribute":true},2019-05-02 10:11:32,29,19,\\N,4,pogs3.3,\\N,RadioField
    2290,TASK_ATTRIBUTE,{"attributeStringValue":"4 - Fairly Well"||"attributeDoubleValue":0.0||"attributeName":"surveyAnswer1"||"attributeIntegerValue":-1||"loggableAttribute":true},2019-05-02 10:11:42,31,17,\\N,4,pogs3.1,\\N,RadioField
    2291,TASK_ATTRIBUTE,{"attributeStringValue":"5 - Very Well"||"attributeDoubleValue":0.0||"attributeName":"surveyAnswer2"||"attributeIntegerValue":-1||"loggableAttribute":true},2019-05-02 10:11:42,28,18,\\N,4,pogs3.2,\\N,RadioField
    2292,TASK_ATTRIBUTE,{"attributeStringValue":"4 - Fairly Well"||"attributeDoubleValue":0.0||"attributeName":"surveyAnswer2"||"attributeIntegerValue":-1||"loggableAttribute":true},2019-05-02 10:11:42,28,18,\\N,4,pogs3.2,\\N,RadioField
    2293,TASK_ATTRIBUTE,{"attributeStringValue":"3 - Somewhat"||"attributeDoubleValue":0.0||"attributeName":"surveyAnswer2"||"attributeIntegerValue":-1||"loggableAttribute":true},2019-05-02 10:11:43,28,18,\\N,4,pogs3.2,\\N,RadioField
    2294,TASK_ATTRIBUTE,{"attributeStringValue":"2 - Slightly"||"attributeDoubleValue":0.0||"attributeName":"surveyAnswer2"||"attributeIntegerValue":-1||"loggableAttribute":true},2019-05-02 10:11:43,28,18,\\N,4,pogs3.2,\\N,RadioField
    2295,TASK_ATTRIBUTE,{"attributeStringValue":"4 - Fairly Well"||"attributeDoubleValue":0.0||"attributeName":"surveyAnswer2"||"attributeIntegerValue":-1||"loggableAttribute":true},2019-05-02 10:11:44,31,17,\\N,4,pogs3.1,\\N,RadioField
    2296,TASK_ATTRIBUTE,{"attributeStringValue":"4 - Fairly Well"||"attributeDoubleValue":0.0||"attributeName":"surveyAnswer0"||"attributeIntegerValue":-1||"loggableAttribute":true},2019-05-02 10:11:48,31,17,\\N,4,pogs3.1,\\N,RadioField
    2297,TASK_ATTRIBUTE,{"attributeStringValue":"2 - Slightly"||"attributeDoubleValue":0.0||"attributeName":"surveyAnswer1"||"attributeIntegerValue":-1||"loggableAttribute":true},2019-05-02 10:11:51,28,18,\\N,4,pogs3.2,\\N,RadioField
    2298,COMMUNICATION_MESSAGE,{"channel":null||"message":""||"type":"JOINED"},2019-05-02 10:12:09,32,20,\\N,4,pogs3.4,\\N,\\N
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
    2347,TASK_ATTRIBUTE,{"attributeStringValue":"Agent Ratings pogs3.2=0+pogs3.4=0+pogs3.3=0+pogs3.1=0 Member Influences pogs3.2=2+pogs3.4=1+pogs3.3=3+pogs3.1=94"||"attributeDoubleValue":0.0||"attributeName":"jeopardyAnswer0__pogs3.1"||"attributeIntegerValue":1||"loggableAttribute":true},2019-05-02 10:23:08,32,17,\\N,4,pogs3.1,\\N,InfluenceMatrix
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
    2365,TASK_ATTRIBUTE,{"attributeStringValue":"Tanzania"||"attributeDoubleValue":29.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":4||"loggableAttribute":true},5/2/19 10:24,32,20,\\N,4,pogs3.4,\\N,IndividualResponse
    2366,TASK_ATTRIBUTE,{"attributeStringValue":"Tanzania"||"attributeDoubleValue":29.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":4||"loggableAttribute":true},5/2/19 10:24,32,19,\\N,4,pogs3.3,\\N,IndividualResponse
    2367,TASK_ATTRIBUTE,{"attributeStringValue":"Malaysia"||"attributeDoubleValue":29.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":4||"loggableAttribute":true},5/2/19 10:24,32,18,\\N,4,pogs3.2,\\N,IndividualResponse
    2368,TASK_ATTRIBUTE,{"attributeStringValue":"Thailand"||"attributeDoubleValue":29.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":4||"loggableAttribute":true},5/2/19 10:25,32,17,\\N,4,pogs3.1,\\N,IndividualResponse
    2369,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"its thailand"||"type":"MESSAGE"},5/2/19 10:25,32,17,\\N,4,pogs3.1,\\N,\\N
    2370,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"I have 0 clue"||"type":"MESSAGE"},5/2/19 10:25,32,20,\\N,4,pogs3.4,\\N,\\N
    2371,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"i have no idea"||"type":"MESSAGE"},5/2/19 10:25,32,19,\\N,4,pogs3.3,\\N,\\N
    2372,TASK_ATTRIBUTE,{"attributeStringValue":"Thailand"||"attributeDoubleValue":29.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":4||"loggableAttribute":true},5/2/19 10:25,32,18,\\N,4,pogs3.2,\\N,GroupRadioResponse
    2373,TASK_ATTRIBUTE,{"attributeStringValue":"Thailand"||"attributeDoubleValue":29.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":4||"loggableAttribute":true},5/2/19 10:25,32,19,\\N,4,pogs3.3,\\N,GroupRadioResponse
    2374,TASK_ATTRIBUTE,{"attributeStringValue":"Thailand"||"attributeDoubleValue":29.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":4||"loggableAttribute":true},5/2/19 10:25,32,20,\\N,4,pogs3.4,\\N,GroupRadioResponse
    2375,TASK_ATTRIBUTE,{"attributeStringValue":"Thailand"||"attributeDoubleValue":29.0||"attributeName":"jeopardyAnswer0__pogs3.3"||"attributeIntegerValue":9||"loggableAttribute":true},5/2/19 10:25,32,19,\\N,4,pogs3.3,\\N,SubmitButtonField
    2376,TASK_ATTRIBUTE,{"attributeStringValue":"Thailand"||"attributeDoubleValue":29.0||"attributeName":"jeopardyAnswer0__pogs3.3"||"attributeIntegerValue":14||"loggableAttribute":true},5/2/19 10:25,32,19,\\N,4,pogs3.3,\\N,SubmitButtonField
    2377,TASK_ATTRIBUTE,{"attributeStringValue":"French Revolution"||"attributeDoubleValue":32.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":9||"loggableAttribute":true},5/2/19 10:26,32,20,\\N,4,pogs3.4,\\N,IndividualResponse
    2378,TASK_ATTRIBUTE,{"attributeStringValue":"French Revolution"||"attributeDoubleValue":32.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":9||"loggableAttribute":true},5/2/19 10:26,32,18,\\N,4,pogs3.2,\\N,IndividualResponse
    2379,TASK_ATTRIBUTE,{"attributeStringValue":"French Revolution"||"attributeDoubleValue":32.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":9||"loggableAttribute":true},5/2/19 10:26,32,17,\\N,4,pogs3.1,\\N,IndividualResponse
    2380,TASK_ATTRIBUTE,{"attributeStringValue":"American Revolution"||"attributeDoubleValue":32.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":9||"loggableAttribute":true},5/2/19 10:26,32,19,\\N,4,pogs3.3,\\N,IndividualResponse
    2381,TASK_ATTRIBUTE,{"attributeStringValue":"French Revolution"||"attributeDoubleValue":32.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":9||"loggableAttribute":true},5/2/19 10:26,32,19,\\N,4,pogs3.3,\\N,GroupRadioResponse
    2382,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"again; no clue"||"type":"MESSAGE"},5/2/19 10:26,32,20,\\N,4,pogs3.4,\\N,\\N
    2383,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"I never read the book"||"type":"MESSAGE"},5/2/19 10:26,32,20,\\N,4,pogs3.4,\\N,\\N
    2384,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"i guessed as well"||"type":"MESSAGE"},5/2/19 10:26,32,18,\\N,4,pogs3.2,\\N,\\N
    2385,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"lol pretty sure its french revolution"||"type":"MESSAGE"},5/2/19 10:26,32,17,\\N,4,pogs3.1,\\N,\\N
    2386,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"me either"||"type":"MESSAGE"},5/2/19 10:26,32,19,\\N,4,pogs3.3,\\N,\\N
    2387,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"Did any of you read it?"||"type":"MESSAGE"},5/2/19 10:26,32,20,\\N,4,pogs3.4,\\N,\\N
    2388,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"Should we ask the machine?"||"type":"MESSAGE"},5/2/19 10:27,32,20,\\N,4,pogs3.4,\\N,\\N
    2389,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"he had a tendency to use french history"||"type":"MESSAGE"},5/2/19 10:27,32,17,\\N,4,pogs3.1,\\N,\\N
    2390,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"no"||"type":"MESSAGE"},5/2/19 10:27,32,18,\\N,4,pogs3.2,\\N,\\N
    2391,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"ima just go with the majority"||"type":"MESSAGE"},5/2/19 10:27,32,19,\\N,4,pogs3.3,\\N,\\N
    2392,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"I already used mine"||"type":"MESSAGE"},5/2/19 10:27,32,20,\\N,4,pogs3.4,\\N,\\N
    2393,TASK_ATTRIBUTE,{"attributeStringValue":"French Revolution"||"attributeDoubleValue":32.0||"attributeName":"jeopardyAnswer0__pogs3.1"||"attributeIntegerValue":14||"loggableAttribute":true},5/2/19 10:27,32,17,\\N,4,pogs3.1,\\N,SubmitButtonField
    2394,TASK_ATTRIBUTE,{"attributeStringValue":"Dwight D Eisenhower"||"attributeDoubleValue":27.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":14||"loggableAttribute":true},5/2/19 10:27,32,17,\\N,4,pogs3.1,\\N,IndividualResponse
    2395,TASK_ATTRIBUTE,{"attributeStringValue":"Richard Nixon"||"attributeDoubleValue":27.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":14||"loggableAttribute":true},5/2/19 10:27,32,20,\\N,4,pogs3.4,\\N,IndividualResponse
    2396,TASK_ATTRIBUTE,{"attributeStringValue":"Richard Nixon"||"attributeDoubleValue":27.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":14||"loggableAttribute":true},5/2/19 10:27,32,18,\\N,4,pogs3.2,\\N,IndividualResponse
    2397,TASK_ATTRIBUTE,{"attributeStringValue":"Harry Truman"||"attributeDoubleValue":27.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":14||"loggableAttribute":true},5/2/19 10:27,32,19,\\N,4,pogs3.3,\\N,IndividualResponse
    2398,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"is it nixon"||"type":"MESSAGE"},5/2/19 10:28,32,19,\\N,4,pogs3.3,\\N,\\N
    2399,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"no eisenhower i think"||"type":"MESSAGE"},5/2/19 10:28,32,17,\\N,4,pogs3.1,\\N,\\N
    2400,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"I'm not that great at history so I guessed"||"type":"MESSAGE"},5/2/19 10:28,32,20,\\N,4,pogs3.4,\\N,\\N
    2401,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"i gussed"||"type":"MESSAGE"},5/2/19 10:28,32,18,\\N,4,pogs3.2,\\N,\\N
    2402,TASK_ATTRIBUTE,{"attributeStringValue":"Dwight D Eisenhower"||"attributeDoubleValue":27.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":14||"loggableAttribute":true},5/2/19 10:28,32,18,\\N,4,pogs3.2,\\N,GroupRadioResponse
    2403,TASK_ATTRIBUTE,{"attributeStringValue":"Dwight D Eisenhower"||"attributeDoubleValue":27.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":14||"loggableAttribute":true},5/2/19 10:28,32,20,\\N,4,pogs3.4,\\N,GroupRadioResponse
    2404,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"nixon resigned"||"type":"MESSAGE"},5/2/19 10:28,32,19,\\N,4,pogs3.3,\\N,\\N
    2405,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"he did watergate"||"type":"MESSAGE"},5/2/19 10:28,32,19,\\N,4,pogs3.3,\\N,\\N
    2406,COMMUNICATION_MESSAGE,{"channel":"group"||"message":""||"type":"MESSAGE"},5/2/19 10:28,32,19,\\N,4,pogs3.3,\\N,\\N
    2407,COMMUNICATION_MESSAGE,{"channel":"group"||"message":""||"type":"MESSAGE"},5/2/19 10:28,32,19,\\N,4,pogs3.3,\\N,\\N
    2408,COMMUNICATION_MESSAGE,{"channel":"group"||"message":""||"type":"MESSAGE"},5/2/19 10:28,32,19,\\N,4,pogs3.3,\\N,\\N
    2409,COMMUNICATION_MESSAGE,{"channel":"group"||"message":""||"type":"MESSAGE"},5/2/19 10:28,32,19,\\N,4,pogs3.3,\\N,\\N
    2410,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"im pretty positive its eisenhower"||"type":"MESSAGE"},5/2/19 10:28,32,17,\\N,4,pogs3.1,\\N,\\N
    2411,COMMUNICATION_MESSAGE,{"channel":"group"||"message":""||"type":"MESSAGE"},5/2/19 10:28,32,19,\\N,4,pogs3.3,\\N,\\N
    2412,TASK_ATTRIBUTE,{"attributeStringValue":"Dwight D Eisenhower"||"attributeDoubleValue":27.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":14||"loggableAttribute":true},5/2/19 10:28,32,19,\\N,4,pogs3.3,\\N,GroupRadioResponse
    2413,TASK_ATTRIBUTE,{"attributeStringValue":"Dwight D Eisenhower"||"attributeDoubleValue":27.0||"attributeName":"jeopardyAnswer0__pogs3.4"||"attributeIntegerValue":19||"loggableAttribute":true},5/2/19 10:28,32,20,\\N,4,pogs3.4,\\N,SubmitButtonField
    2414,TASK_ATTRIBUTE,{"attributeStringValue":"Dwight D Eisenhower"||"attributeDoubleValue":27.0||"attributeName":"jeopardyAnswer0__pogs3.3"||"attributeIntegerValue":19||"loggableAttribute":true},5/2/19 10:28,32,19,\\N,4,pogs3.3,\\N,SubmitButtonField
    2415,TASK_ATTRIBUTE,{"attributeStringValue":"Dell"||"attributeDoubleValue":6.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":19||"loggableAttribute":true},5/2/19 10:28,32,17,\\N,4,pogs3.1,\\N,IndividualResponse
    2416,TASK_ATTRIBUTE,{"attributeStringValue":"HP"||"attributeDoubleValue":6.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":19||"loggableAttribute":true},5/2/19 10:29,32,18,\\N,4,pogs3.2,\\N,IndividualResponse
    2417,TASK_ATTRIBUTE,{"attributeStringValue":"Dell"||"attributeDoubleValue":6.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":19||"loggableAttribute":true},5/2/19 10:29,32,20,\\N,4,pogs3.4,\\N,IndividualResponse
    2418,TASK_ATTRIBUTE,{"attributeStringValue":"Dell"||"attributeDoubleValue":6.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":19||"loggableAttribute":true},5/2/19 10:29,32,19,\\N,4,pogs3.3,\\N,IndividualResponse
    2419,TASK_ATTRIBUTE,{"attributeStringValue":"Google"||"attributeDoubleValue":6.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":19||"loggableAttribute":true},5/2/19 10:29,32,18,\\N,4,pogs3.2,\\N,IndividualResponse
    2420,TASK_ATTRIBUTE,{"attributeStringValue":"Dell"||"attributeDoubleValue":6.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":19||"loggableAttribute":true},5/2/19 10:29,32,18,\\N,4,pogs3.2,\\N,GroupRadioResponse
    2421,TASK_ATTRIBUTE,{"attributeStringValue":"Dell"||"attributeDoubleValue":6.0||"attributeName":"jeopardyAnswer0__pogs3.1"||"attributeIntegerValue":19||"loggableAttribute":true},5/2/19 10:30,32,17,\\N,4,pogs3.1,\\N,SubmitButtonField
    2422,TASK_ATTRIBUTE,{"attributeStringValue":"Dell"||"attributeDoubleValue":6.0||"attributeName":"jeopardyAnswer0__pogs3.3"||"attributeIntegerValue":19||"loggableAttribute":true},5/2/19 10:30,32,19,\\N,4,pogs3.3,\\N,SubmitButtonField
    2423,TASK_ATTRIBUTE,{"attributeStringValue":"Agent Ratings pogs3.2=1+pogs3.4=1+pogs3.3=1+pogs3.1=1 Member Influences pogs3.2=1+pogs3.4=1+pogs3.3=1+pogs3.1=97"||"attributeDoubleValue":0.0||"attributeName":"jeopardyAnswer0__pogs3.1"||"attributeIntegerValue":2||"loggableAttribute":true},5/2/19 10:31,32,17,\\N,4,pogs3.1,\\N,InfluenceMatrix
    2424,TASK_ATTRIBUTE,{"attributeStringValue":"Agent Ratings pogs3.2=2+pogs3.4=2+pogs3.3=4+pogs3.1=4 Member Influences pogs3.2=10+pogs3.4=20+pogs3.3=30+pogs3.1=40"||"attributeDoubleValue":0.0||"attributeName":"jeopardyAnswer0__pogs3.1"||"attributeIntegerValue":2||"loggableAttribute":true},5/2/19 10:31,32,17,\\N,4,pogs3.2,\\N,InfluenceMatrix
    2425,TASK_ATTRIBUTE,{"attributeStringValue":"Agent Ratings pogs3.2=0+pogs3.4=0+pogs3.3=0+pogs3.1=0 Member Influences pogs3.2=5+pogs3.4=20+pogs3.3=5+pogs3.1=70"||"attributeDoubleValue":0.0||"attributeName":"jeopardyAnswer0__pogs3.3"||"attributeIntegerValue":2||"loggableAttribute":true},5/2/19 10:31,32,19,\\N,4,pogs3.3,\\N,InfluenceMatrix
    2426,TASK_ATTRIBUTE,{"attributeStringValue":"Agent Ratings pogs3.2=4+pogs3.4=4+pogs3.3=2+pogs3.1=2 Member Influences pogs3.2=40+pogs3.4=30+pogs3.3=20+pogs3.1=10"||"attributeDoubleValue":0.0||"attributeName":"jeopardyAnswer0__pogs3.1"||"attributeIntegerValue":2||"loggableAttribute":true},5/2/19 10:31,32,17,\\N,4,pogs3.4,\\N,InfluenceMatrix
    2427,TASK_ATTRIBUTE,{"attributeStringValue":"Manticore"||"attributeDoubleValue":24.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":19||"loggableAttribute":true},5/2/19 10:31,32,20,\\N,4,pogs3.4,\\N,IndividualResponse
    2428,TASK_ATTRIBUTE,{"attributeStringValue":"Chimera"||"attributeDoubleValue":24.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":19||"loggableAttribute":true},5/2/19 10:31,32,18,\\N,4,pogs3.2,\\N,IndividualResponse
    2429,TASK_ATTRIBUTE,{"attributeStringValue":"Chimera"||"attributeDoubleValue":24.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":19||"loggableAttribute":true},5/2/19 10:31,32,17,\\N,4,pogs3.1,\\N,IndividualResponse
    2430,TASK_ATTRIBUTE,{"attributeStringValue":"Griffin"||"attributeDoubleValue":24.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":19||"loggableAttribute":true},5/2/19 10:31,32,18,\\N,4,pogs3.2,\\N,IndividualResponse
    2431,TASK_ATTRIBUTE,{"attributeStringValue":"Manticore"||"attributeDoubleValue":24.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":19||"loggableAttribute":true},5/2/19 10:31,32,19,\\N,4,pogs3.3,\\N,IndividualResponse
    2432,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"does anyone know forsure"||"type":"MESSAGE"},5/2/19 10:32,32,19,\\N,4,pogs3.3,\\N,\\N
    2433,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"no"||"type":"MESSAGE"},5/2/19 10:32,32,18,\\N,4,pogs3.2,\\N,\\N
    2434,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"no "||"type":"MESSAGE"},5/2/19 10:32,32,20,\\N,4,pogs3.4,\\N,\\N
    2435,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"i think its chimera"||"type":"MESSAGE"},5/2/19 10:32,32,17,\\N,4,pogs3.1,\\N,\\N
    2436,TASK_ATTRIBUTE,{"attributeStringValue":"Chimera"||"attributeDoubleValue":24.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":19||"loggableAttribute":true},5/2/19 10:32,32,20,\\N,4,pogs3.4,\\N,GroupRadioResponse
    2437,TASK_ATTRIBUTE,{"attributeStringValue":"Chimera"||"attributeDoubleValue":24.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":19||"loggableAttribute":true},5/2/19 10:32,32,18,\\N,4,pogs3.2,\\N,GroupRadioResponse
    2438,TASK_ATTRIBUTE,{"attributeStringValue":"Chimera"||"attributeDoubleValue":24.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":19||"loggableAttribute":true},5/2/19 10:32,32,19,\\N,4,pogs3.3,\\N,GroupRadioResponse
    2439,TASK_ATTRIBUTE,{"attributeStringValue":"Chimera"||"attributeDoubleValue":24.0||"attributeName":"jeopardyAnswer0__pogs3.4"||"attributeIntegerValue":19||"loggableAttribute":true},5/2/19 10:32,32,20,\\N,4,pogs3.4,\\N,SubmitButtonField
    2440,TASK_ATTRIBUTE,{"attributeStringValue":"Switzerland"||"attributeDoubleValue":44.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":19||"loggableAttribute":true},5/2/19 10:32,32,17,\\N,4,pogs3.1,\\N,IndividualResponse
    2441,TASK_ATTRIBUTE,{"attributeStringValue":"France"||"attributeDoubleValue":44.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":19||"loggableAttribute":true},5/2/19 10:32,32,20,\\N,4,pogs3.4,\\N,IndividualResponse
    2442,TASK_ATTRIBUTE,{"attributeStringValue":"England"||"attributeDoubleValue":44.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":19||"loggableAttribute":true},5/2/19 10:32,32,18,\\N,4,pogs3.2,\\N,IndividualResponse
    2443,TASK_ATTRIBUTE,{"attributeStringValue":"France"||"attributeDoubleValue":44.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":19||"loggableAttribute":true},5/2/19 10:33,32,19,\\N,4,pogs3.3,\\N,IndividualResponse
    2444,TASK_ATTRIBUTE,{"attributeStringValue":"France"||"attributeDoubleValue":44.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":19||"loggableAttribute":true},5/2/19 10:33,32,17,\\N,4,pogs3.1,\\N,GroupRadioResponse
    2445,TASK_ATTRIBUTE,{"attributeStringValue":"France"||"attributeDoubleValue":44.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":19||"loggableAttribute":true},5/2/19 10:33,32,18,\\N,4,pogs3.2,\\N,GroupRadioResponse
    2446,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"i think its france because of enlightment"||"type":"MESSAGE"},5/2/19 10:33,32,19,\\N,4,pogs3.3,\\N,\\N
    2447,TASK_ATTRIBUTE,{"attributeStringValue":"France"||"attributeDoubleValue":44.0||"attributeName":"jeopardyAnswer0__pogs3.2"||"attributeIntegerValue":24||"loggableAttribute":true},5/2/19 10:34,32,18,\\N,4,pogs3.2,\\N,SubmitButtonField
    2448,TASK_ATTRIBUTE,{"attributeStringValue":"France"||"attributeDoubleValue":44.0||"attributeName":"jeopardyAnswer0__pogs3.3"||"attributeIntegerValue":24||"loggableAttribute":true},5/2/19 10:34,32,19,\\N,4,pogs3.3,\\N,SubmitButtonField
    2449,TASK_ATTRIBUTE,{"attributeStringValue":"Anubis"||"attributeDoubleValue":30.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":24||"loggableAttribute":true},5/2/19 10:34,32,17,\\N,4,pogs3.1,\\N,IndividualResponse
    2450,TASK_ATTRIBUTE,{"attributeStringValue":"Anubis"||"attributeDoubleValue":30.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":24||"loggableAttribute":true},5/2/19 10:34,32,20,\\N,4,pogs3.4,\\N,IndividualResponse
    2451,TASK_ATTRIBUTE,{"attributeStringValue":"Anubis"||"attributeDoubleValue":30.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":24||"loggableAttribute":true},5/2/19 10:34,32,18,\\N,4,pogs3.2,\\N,IndividualResponse
    2452,TASK_ATTRIBUTE,{"attributeStringValue":"Horus"||"attributeDoubleValue":30.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":24||"loggableAttribute":true},5/2/19 10:34,32,19,\\N,4,pogs3.3,\\N,IndividualResponse
    2453,TASK_ATTRIBUTE,{"attributeStringValue":"Anubis"||"attributeDoubleValue":30.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":24||"loggableAttribute":true},5/2/19 10:34,32,19,\\N,4,pogs3.3,\\N,GroupRadioResponse
    2454,TASK_ATTRIBUTE,{"attributeStringValue":"Anubis"||"attributeDoubleValue":30.0||"attributeName":"jeopardyAnswer0__pogs3.1"||"attributeIntegerValue":29||"loggableAttribute":true},5/2/19 10:35,32,17,\\N,4,pogs3.1,\\N,SubmitButtonField
    2455,TASK_ATTRIBUTE,{"attributeStringValue":"Ringo"||"attributeDoubleValue":33.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":29||"loggableAttribute":true},5/2/19 10:35,32,18,\\N,4,pogs3.2,\\N,IndividualResponse
    2456,TASK_ATTRIBUTE,{"attributeStringValue":"Ringo"||"attributeDoubleValue":33.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":29||"loggableAttribute":true},5/2/19 10:35,32,20,\\N,4,pogs3.4,\\N,IndividualResponse
    2457,TASK_ATTRIBUTE,{"attributeStringValue":"James"||"attributeDoubleValue":33.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":29||"loggableAttribute":true},5/2/19 10:35,32,17,\\N,4,pogs3.1,\\N,IndividualResponse
    2458,TASK_ATTRIBUTE,{"attributeStringValue":"Danger"||"attributeDoubleValue":33.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":29||"loggableAttribute":true},5/2/19 10:35,32,19,\\N,4,pogs3.3,\\N,IndividualResponse
    2459,TASK_ATTRIBUTE,{"attributeStringValue":"Danger"||"attributeDoubleValue":33.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":29||"loggableAttribute":true},5/2/19 10:35,32,17,\\N,4,pogs3.1,\\N,GroupRadioResponse
    2460,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"lol isn't it danger?"||"type":"MESSAGE"},5/2/19 10:36,32,17,\\N,4,pogs3.1,\\N,\\N
    2461,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"i haven't seen it in a while"||"type":"MESSAGE"},5/2/19 10:36,32,17,\\N,4,pogs3.1,\\N,\\N
    2462,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"i haven't watched the movie"||"type":"MESSAGE"},5/2/19 10:36,32,18,\\N,4,pogs3.2,\\N,\\N
    2463,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"I've never watched it"||"type":"MESSAGE"},5/2/19 10:36,32,20,\\N,4,pogs3.4,\\N,\\N
    2464,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"thats what i think i forgott"||"type":"MESSAGE"},5/2/19 10:36,32,19,\\N,4,pogs3.3,\\N,\\N
    2465,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"i think its danger..."||"type":"MESSAGE"},5/2/19 10:36,32,17,\\N,4,pogs3.1,\\N,\\N
    2466,TASK_ATTRIBUTE,{"attributeStringValue":"Danger"||"attributeDoubleValue":33.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":29||"loggableAttribute":true},5/2/19 10:36,32,18,\\N,4,pogs3.2,\\N,GroupRadioResponse
    2467,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"im trynna think of his catch phrases"||"type":"MESSAGE"},5/2/19 10:36,32,19,\\N,4,pogs3.3,\\N,\\N
    2468,TASK_ATTRIBUTE,{"attributeStringValue":"Danger"||"attributeDoubleValue":33.0||"attributeName":"jeopardyAnswer0__pogs3.3"||"attributeIntegerValue":29||"loggableAttribute":true},5/2/19 10:36,32,19,\\N,4,pogs3.3,\\N,SubmitButtonField
    2469,TASK_ATTRIBUTE,{"attributeStringValue":"oxidized copper"||"attributeDoubleValue":14.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":29||"loggableAttribute":true},5/2/19 10:36,32,17,\\N,4,pogs3.1,\\N,IndividualResponse
    2470,TASK_ATTRIBUTE,{"attributeStringValue":"oxidized copper"||"attributeDoubleValue":14.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":29||"loggableAttribute":true},5/2/19 10:36,32,20,\\N,4,pogs3.4,\\N,IndividualResponse
    2471,TASK_ATTRIBUTE,{"attributeStringValue":"oxidized copper"||"attributeDoubleValue":14.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":29||"loggableAttribute":true},5/2/19 10:36,32,18,\\N,4,pogs3.2,\\N,IndividualResponse
    2472,TASK_ATTRIBUTE,{"attributeStringValue":"oxidized copper"||"attributeDoubleValue":14.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":29||"loggableAttribute":true},5/2/19 10:36,32,19,\\N,4,pogs3.3,\\N,IndividualResponse
    2473,TASK_ATTRIBUTE,{"attributeStringValue":"oxidized copper"||"attributeDoubleValue":14.0||"attributeName":"jeopardyAnswer0__pogs3.3"||"attributeIntegerValue":34||"loggableAttribute":true},5/2/19 10:38,32,19,\\N,4,pogs3.3,\\N,SubmitButtonField
    2474,TASK_ATTRIBUTE,{"attributeStringValue":"Agent Ratings pogs3.2=1+pogs3.4=1+pogs3.3=1+pogs3.1=1 Member Influences pogs3.2=1+pogs3.4=1+pogs3.3=1+pogs3.1=97"||"attributeDoubleValue":0.0||"attributeName":"jeopardyAnswer0__pogs3.1"||"attributeIntegerValue":3||"loggableAttribute":true},5/2/19 10:38,32,17,\\N,4,pogs3.1,\\N,InfluenceMatrix
    2475,TASK_ATTRIBUTE,{"attributeStringValue":"Agent Ratings pogs3.2=1+pogs3.4=1+pogs3.3=1+pogs3.1=1 Member Influences pogs3.2=1+pogs3.4=1+pogs3.3=1+pogs3.1=97"||"attributeDoubleValue":0.0||"attributeName":"jeopardyAnswer0__pogs3.1"||"attributeIntegerValue":3||"loggableAttribute":true},5/2/19 10:38,32,17,\\N,4,pogs3.1,\\N,InfluenceMatrix
    2476,TASK_ATTRIBUTE,{"attributeStringValue":"Agent Ratings pogs3.2=0+pogs3.4=1+pogs3.3=0+pogs3.1=0 Member Influences pogs3.2=10+pogs3.4=20+pogs3.3=10+pogs3.1=60"||"attributeDoubleValue":0.0||"attributeName":"jeopardyAnswer0__pogs3.3"||"attributeIntegerValue":3||"loggableAttribute":true},5/2/19 10:38,32,19,\\N,4,pogs3.3,\\N,InfluenceMatrix
    2477,TASK_ATTRIBUTE,{"attributeStringValue":"Agent Ratings pogs3.2=1+pogs3.4=1+pogs3.3=1+pogs3.1=1 Member Influences pogs3.2=1+pogs3.4=1+pogs3.3=1+pogs3.1=97"||"attributeDoubleValue":0.0||"attributeName":"jeopardyAnswer0__pogs3.1"||"attributeIntegerValue":3||"loggableAttribute":true},5/2/19 10:38,32,17,\\N,4,pogs3.1,\\N,InfluenceMatrix
    2478,TASK_ATTRIBUTE,{"attributeStringValue":"Agent Ratings pogs3.2=0+pogs3.4=1+pogs3.3=0+pogs3.1=0 Member Influences pogs3.2=10+pogs3.4=20+pogs3.3=10+pogs3.1=60"||"attributeDoubleValue":0.0||"attributeName":"jeopardyAnswer0__pogs3.3"||"attributeIntegerValue":3||"loggableAttribute":true},5/2/19 10:38,32,19,\\N,4,pogs3.3,\\N,InfluenceMatrix
    2479,TASK_ATTRIBUTE,{"attributeStringValue":"Truman Capote"||"attributeDoubleValue":37.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":29||"loggableAttribute":true},5/2/19 10:39,32,19,\\N,4,pogs3.3,\\N,IndividualResponse
    2480,TASK_ATTRIBUTE,{"attributeStringValue":"Kurt Vonnegut"||"attributeDoubleValue":37.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":29||"loggableAttribute":true},5/2/19 10:39,32,18,\\N,4,pogs3.2,\\N,IndividualResponse
    2481,TASK_ATTRIBUTE,{"attributeStringValue":"Truman Capote"||"attributeDoubleValue":37.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":29||"loggableAttribute":true},5/2/19 10:39,32,20,\\N,4,pogs3.4,\\N,IndividualResponse
    2482,TASK_ATTRIBUTE,{"attributeStringValue":"Mario Puzo"||"attributeDoubleValue":37.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":29||"loggableAttribute":true},5/2/19 10:39,32,17,\\N,4,pogs3.1,\\N,IndividualResponse
    2483,TASK_ATTRIBUTE,{"attributeStringValue":"Mario Puzo"||"attributeDoubleValue":37.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":29||"loggableAttribute":true},5/2/19 10:39,32,19,\\N,4,pogs3.3,\\N,IndividualResponse
    2484,TASK_ATTRIBUTE,{"attributeStringValue":"Truman Capote"||"attributeDoubleValue":37.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":29||"loggableAttribute":true},5/2/19 10:39,32,17,\\N,4,pogs3.1,\\N,GroupRadioResponse
    2485,TASK_ATTRIBUTE,{"attributeStringValue":"Mario Puzo"||"attributeDoubleValue":37.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":29||"loggableAttribute":true},5/2/19 10:39,32,17,\\N,4,pogs3.1,\\N,GroupRadioResponse
    2486,TASK_ATTRIBUTE,{"attributeStringValue":"Truman Capote"||"attributeDoubleValue":37.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":29||"loggableAttribute":true},5/2/19 10:39,32,19,\\N,4,pogs3.3,\\N,GroupRadioResponse
    2487,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"anyone know for sure?"||"type":"MESSAGE"},5/2/19 10:40,32,20,\\N,4,pogs3.4,\\N,\\N
    2488,TASK_ATTRIBUTE,{"attributeStringValue":"Truman Capote"||"attributeDoubleValue":37.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":29||"loggableAttribute":true},5/2/19 10:40,32,18,\\N,4,pogs3.2,\\N,GroupRadioResponse
    2489,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"95% sure its puzo"||"type":"MESSAGE"},5/2/19 10:40,32,17,\\N,4,pogs3.1,\\N,\\N
    2490,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"im stuck on the last 2"||"type":"MESSAGE"},5/2/19 10:40,32,19,\\N,4,pogs3.3,\\N,\\N
    2491,TASK_ATTRIBUTE,{"attributeStringValue":"Mario Puzo"||"attributeDoubleValue":37.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":29||"loggableAttribute":true},5/2/19 10:40,32,18,\\N,4,pogs3.2,\\N,GroupRadioResponse
    2492,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"puzo sounds more italian"||"type":"MESSAGE"},5/2/19 10:40,32,19,\\N,4,pogs3.3,\\N,\\N
    2493,TASK_ATTRIBUTE,{"attributeStringValue":"Mario Puzo"||"attributeDoubleValue":37.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":29||"loggableAttribute":true},5/2/19 10:40,32,19,\\N,4,pogs3.3,\\N,GroupRadioResponse
    2494,TASK_ATTRIBUTE,{"attributeStringValue":"Mario Puzo"||"attributeDoubleValue":37.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":29||"loggableAttribute":true},5/2/19 10:40,32,20,\\N,4,pogs3.4,\\N,GroupRadioResponse
    2495,TASK_ATTRIBUTE,{"attributeStringValue":"Mario Puzo"||"attributeDoubleValue":37.0||"attributeName":"jeopardyAnswer0__pogs3.3"||"attributeIntegerValue":34||"loggableAttribute":true},5/2/19 10:40,32,19,\\N,4,pogs3.3,\\N,SubmitButtonField
    2496,TASK_ATTRIBUTE,{"attributeStringValue":"Mt. Vesuvius"||"attributeDoubleValue":23.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":34||"loggableAttribute":true},5/2/19 10:40,32,20,\\N,4,pogs3.4,\\N,IndividualResponse
    2497,TASK_ATTRIBUTE,{"attributeStringValue":"Mt. Vesuvius"||"attributeDoubleValue":23.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":34||"loggableAttribute":true},5/2/19 10:40,32,17,\\N,4,pogs3.1,\\N,IndividualResponse
    2498,TASK_ATTRIBUTE,{"attributeStringValue":"Mt. Vesuvius"||"attributeDoubleValue":23.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":34||"loggableAttribute":true},5/2/19 10:40,32,19,\\N,4,pogs3.3,\\N,IndividualResponse
    2499,TASK_ATTRIBUTE,{"attributeStringValue":"Mt. St. Helens"||"attributeDoubleValue":23.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":34||"loggableAttribute":true},5/2/19 10:40,32,18,\\N,4,pogs3.2,\\N,IndividualResponse
    2500,TASK_ATTRIBUTE,{"attributeStringValue":"Mt. Vesuvius"||"attributeDoubleValue":23.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":34||"loggableAttribute":true},5/2/19 10:41,32,18,\\N,4,pogs3.2,\\N,GroupRadioResponse
    2501,TASK_ATTRIBUTE,{"attributeStringValue":"Mt. Vesuvius"||"attributeDoubleValue":23.0||"attributeName":"jeopardyAnswer0__pogs3.1"||"attributeIntegerValue":39||"loggableAttribute":true},5/2/19 10:41,32,17,\\N,4,pogs3.1,\\N,SubmitButtonField
    2502,TASK_ATTRIBUTE,{"attributeStringValue":"Mt. Vesuvius"||"attributeDoubleValue":23.0||"attributeName":"jeopardyAnswer0__pogs3.3"||"attributeIntegerValue":39||"loggableAttribute":true},5/2/19 10:41,32,19,\\N,4,pogs3.3,\\N,SubmitButtonField
    2503,TASK_ATTRIBUTE,{"attributeStringValue":"Panama Canal"||"attributeDoubleValue":22.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":39||"loggableAttribute":true},5/2/19 10:41,32,17,\\N,4,pogs3.1,\\N,IndividualResponse
    2504,TASK_ATTRIBUTE,{"attributeStringValue":"Panama Canal"||"attributeDoubleValue":22.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":39||"loggableAttribute":true},5/2/19 10:41,32,18,\\N,4,pogs3.2,\\N,IndividualResponse
    2505,TASK_ATTRIBUTE,{"attributeStringValue":"The Straits of Gibraltar"||"attributeDoubleValue":22.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":39||"loggableAttribute":true},5/2/19 10:41,32,20,\\N,4,pogs3.4,\\N,IndividualResponse
    2506,TASK_ATTRIBUTE,{"attributeStringValue":"The Strait of Magellan"||"attributeDoubleValue":22.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":39||"loggableAttribute":true},5/2/19 10:42,32,19,\\N,4,pogs3.3,\\N,IndividualResponse
    2507,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"panama canal is above south america"||"type":"MESSAGE"},5/2/19 10:42,32,19,\\N,4,pogs3.3,\\N,\\N
    2508,TASK_ATTRIBUTE,{"attributeStringValue":"The Strait of Magellan"||"attributeDoubleValue":22.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":39||"loggableAttribute":true},5/2/19 10:42,32,17,\\N,4,pogs3.1,\\N,GroupRadioResponse
    2509,TASK_ATTRIBUTE,{"attributeStringValue":"The Straits of Gibraltar"||"attributeDoubleValue":22.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":39||"loggableAttribute":true},5/2/19 10:42,32,18,\\N,4,pogs3.2,\\N,GroupRadioResponse
    2510,TASK_ATTRIBUTE,{"attributeStringValue":"The Strait of Magellan"||"attributeDoubleValue":22.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":39||"loggableAttribute":true},5/2/19 10:42,32,18,\\N,4,pogs3.2,\\N,GroupRadioResponse
    2511,TASK_ATTRIBUTE,{"attributeStringValue":"The Strait of Magellan"||"attributeDoubleValue":22.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":39||"loggableAttribute":true},5/2/19 10:42,32,20,\\N,4,pogs3.4,\\N,GroupRadioResponse
    2512,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"idk about the straits"||"type":"MESSAGE"},5/2/19 10:42,32,19,\\N,4,pogs3.3,\\N,\\N
    2513,TASK_ATTRIBUTE,{"attributeStringValue":"The Strait of Magellan"||"attributeDoubleValue":22.0||"attributeName":"jeopardyAnswer0__pogs3.1"||"attributeIntegerValue":44||"loggableAttribute":true},5/2/19 10:43,32,17,\\N,4,pogs3.1,\\N,SubmitButtonField
    2514,TASK_ATTRIBUTE,{"attributeStringValue":"Through the Looking Glass"||"attributeDoubleValue":45.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":44||"loggableAttribute":true},5/2/19 10:43,32,20,\\N,4,pogs3.4,\\N,IndividualResponse
    2515,TASK_ATTRIBUTE,{"attributeStringValue":"Pippi Longstocking"||"attributeDoubleValue":45.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":44||"loggableAttribute":true},5/2/19 10:43,32,19,\\N,4,pogs3.3,\\N,IndividualResponse
    2516,TASK_ATTRIBUTE,{"attributeStringValue":"Alice in Wonderland"||"attributeDoubleValue":45.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":44||"loggableAttribute":true},5/2/19 10:43,32,18,\\N,4,pogs3.2,\\N,IndividualResponse
    2517,TASK_ATTRIBUTE,{"attributeStringValue":"Pippi Longstocking"||"attributeDoubleValue":45.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":44||"loggableAttribute":true},5/2/19 10:43,32,18,\\N,4,pogs3.2,\\N,IndividualResponse
    2518,TASK_ATTRIBUTE,{"attributeStringValue":"Through the Looking Glass"||"attributeDoubleValue":45.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":44||"loggableAttribute":true},5/2/19 10:43,32,17,\\N,4,pogs3.1,\\N,IndividualResponse
    2519,TASK_ATTRIBUTE,{"attributeStringValue":"Alice in Wonderland"||"attributeDoubleValue":45.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":44||"loggableAttribute":true},5/2/19 10:43,32,18,\\N,4,pogs3.2,\\N,IndividualResponse
    2520,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"i have no clue"||"type":"MESSAGE"},5/2/19 10:44,32,19,\\N,4,pogs3.3,\\N,\\N
    2521,TASK_ATTRIBUTE,{"attributeStringValue":"Through the Looking Glass"||"attributeDoubleValue":45.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":44||"loggableAttribute":true},5/2/19 10:44,32,18,\\N,4,pogs3.2,\\N,GroupRadioResponse
    2522,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"its between that and alice in wonderland"||"type":"MESSAGE"},5/2/19 10:44,32,17,\\N,4,pogs3.1,\\N,\\N
    2523,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"idk"||"type":"MESSAGE"},5/2/19 10:44,32,17,\\N,4,pogs3.1,\\N,\\N
    2524,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"I've never read Pippi Longstocking"||"type":"MESSAGE"},5/2/19 10:44,32,20,\\N,4,pogs3.4,\\N,\\N
    2525,TASK_ATTRIBUTE,{"attributeStringValue":"Through the Looking Glass"||"attributeDoubleValue":45.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":44||"loggableAttribute":true},5/2/19 10:44,32,19,\\N,4,pogs3.3,\\N,GroupRadioResponse
    2526,TASK_ATTRIBUTE,{"attributeStringValue":"Through the Looking Glass"||"attributeDoubleValue":45.0||"attributeName":"jeopardyAnswer0__pogs3.1"||"attributeIntegerValue":49||"loggableAttribute":true},5/2/19 10:44,32,17,\\N,4,pogs3.1,\\N,SubmitButtonField
    2527,TASK_ATTRIBUTE,{"attributeStringValue":"Adam and Eve"||"attributeDoubleValue":16.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":49||"loggableAttribute":true},5/2/19 10:44,32,20,\\N,4,pogs3.4,\\N,IndividualResponse
    2528,TASK_ATTRIBUTE,{"attributeStringValue":"Eve and Serpent"||"attributeDoubleValue":16.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":49||"loggableAttribute":true},5/2/19 10:44,32,17,\\N,4,pogs3.1,\\N,IndividualResponse
    2529,TASK_ATTRIBUTE,{"attributeStringValue":"Eve and Serpent"||"attributeDoubleValue":16.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":49||"loggableAttribute":true},5/2/19 10:44,32,18,\\N,4,pogs3.2,\\N,IndividualResponse
    2530,TASK_ATTRIBUTE,{"attributeStringValue":"Eve and Serpent"||"attributeDoubleValue":16.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":49||"loggableAttribute":true},5/2/19 10:44,32,19,\\N,4,pogs3.3,\\N,IndividualResponse
    2531,TASK_ATTRIBUTE,{"attributeStringValue":"Adam and Eve"||"attributeDoubleValue":16.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":49||"loggableAttribute":true},5/2/19 10:44,32,19,\\N,4,pogs3.3,\\N,IndividualResponse
    2532,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"does anyone know"||"type":"MESSAGE"},5/2/19 10:45,32,19,\\N,4,pogs3.3,\\N,\\N
    2533,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"Not for sure "||"type":"MESSAGE"},5/2/19 10:45,32,20,\\N,4,pogs3.4,\\N,\\N
    2534,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"It would make sense for it to be the serpent and eve actually "||"type":"MESSAGE"},5/2/19 10:45,32,20,\\N,4,pogs3.4,\\N,\\N
    2535,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"idk i thought eve and serpant cuz it led to trouble"||"type":"MESSAGE"},5/2/19 10:45,32,17,\\N,4,pogs3.1,\\N,\\N
    2536,TASK_ATTRIBUTE,{"attributeStringValue":"Eve and Serpent"||"attributeDoubleValue":16.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":49||"loggableAttribute":true},5/2/19 10:45,32,19,\\N,4,pogs3.3,\\N,GroupRadioResponse
    2537,COMMUNICATION_MESSAGE,{"channel":"group"||"message":"because Adam and Eve didn't exactly have conversations that lead to trouble "||"type":"MESSAGE"},5/2/19 10:45,32,20,\\N,4,pogs3.4,\\N,\\N
    2538,TASK_ATTRIBUTE,{"attributeStringValue":"Eve and Serpent"||"attributeDoubleValue":16.0||"attributeName":"jeopardyAnswer0"||"attributeIntegerValue":49||"loggableAttribute":true},5/2/19 10:45,32,20,\\N,4,pogs3.4,\\N,GroupRadioResponse
    2539,TASK_ATTRIBUTE,{"attributeStringValue":"Eve and Serpent"||"attributeDoubleValue":16.0||"attributeName":"jeopardyAnswer0__pogs3.1"||"attributeIntegerValue":54||"loggableAttribute":true},5/2/19 10:46,32,17,\\N,4,pogs3.1,\\N,SubmitButtonField
    2540,TASK_ATTRIBUTE,{"attributeStringValue":"Agent Ratings pogs3.2=1+pogs3.4=1+pogs3.3=1+pogs3.1=1 Member Influences pogs3.2=0+pogs3.4=2+pogs3.3=2+pogs3.1=96"||"attributeDoubleValue":0.0||"attributeName":"jeopardyAnswer0__pogs3.1"||"attributeIntegerValue":4||"loggableAttribute":true},5/2/19 10:47,32,17,\\N,4,pogs3.1,\\N,InfluenceMatrix
    2541,TASK_ATTRIBUTE,{"attributeStringValue":"Agent Ratings pogs3.2=0+pogs3.4=1+pogs3.3=0+pogs3.1=0 Member Influences pogs3.2=20+pogs3.4=30+pogs3.3=20+pogs3.1=30"||"attributeDoubleValue":0.0||"attributeName":"jeopardyAnswer0__pogs3.3"||"attributeIntegerValue":4||"loggableAttribute":true},5/2/19 10:47,32,19,\\N,4,pogs3.3,\\N,InfluenceMatrix
    2542,TASK_ATTRIBUTE,{"attributeStringValue":"Agent Ratings pogs3.2=+pogs3.4=+pogs3.3=1+pogs3.1=1 Member Influences pogs3.2=+pogs3.4=+pogs3.3=+pogs3.1="||"attributeDoubleValue":0.0||"attributeName":"jeopardyAnswer0__pogs3.1"||"attributeIntegerValue":4||"loggableAttribute":true},5/2/19 10:47,32,17,\\N,4,pogs3.2,\\N,InfluenceMatrix
'''


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
            "ID":43
            ,"question":"In which Shakespeare play are Stephano and Trinculo characters?"
            ,"Answer":"The Tempest"
            ,"value":["The Tempest", "Hamlet", "As You Like It", "The Merchant of Venice"]
            ,"Category":"Literature and Media"
            ,"Level":"Hard"
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
            "ID":4
            ,"question":"Balloons are filled with"
            ,"Answer":"Helium"
            ,"value":["oxygen", "nitrogen", "argon", "Helium"]
            ,"Category":"Science and Technology"
            ,"Level":"Easy"
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
            "ID":29
            ,"question":"Ayutthaya is the historical former capital city of this modern-day country"
            ,"Answer":"Thailand"
            ,"value":["Malaysia", "Vietnam", "Tanzania", "Thailand"]
            ,"Category":"History and Mythology"
            ,"Level":"Hard"
        },
        {
            "ID":32
            ,"question":"A Tale of Two Cities' by Charles Dickens is based on this historical event."
            ,"Answer":"French Revolution"
            ,"value":["American Revolution", "Chinese Revolution", "Russian Revolution", "French Revolution"]
            ,"Category":"Literature and Media"
            ,"Level":"Easy"
        },
        {
            "ID":27
            ,"question":"This US president authorized the creation of NASA"
            ,"Answer":"Dwight D Eisenhower"
            ,"value":["Richard Nixon", "Harry Truman", "John F Kennedy", "Dwight D Eisenhower"]
            ,"Category":"History and Mythology"
            ,"Level":"Hard"
        },
        {
            "ID":6
            ,"question":"On the verge of going bust in 1997, it was saved by a $150M investment by rival Microsoft"
            ,"Answer":"Apple"
            ,"value":["HP", "Google", "Dell", "Apple"]
            ,"Category":"Science and Technology"
            ,"Level":"Medium"
        },
        {
            "ID":24
            ,"question":"What mythological beast has the head of a man, the body of a lion, and the tail and feet of a dragon?"
            ,"Answer":"Manticore"
            ,"value":["Manticore", "Chimera", "Cyclops", "Griffin"]
            ,"Category":"History and Mythology"
            ,"Level":"Medium"
        },
        {
            "ID":44
            ,"question":"This country's authors have won more Literature Nobel Prizes than any other, including the US"
            ,"Answer":"France"
            ,"value":["France", "England", "Switzerland", "Greece"]
            ,"Category":"Literature and Media"
            ,"Level":"Hard"
        },
        {
            "ID":30
            ,"question":"In ancient Egyption mythology, who supervised the weighing of souls at judgement?"
            ,"Answer":"Anubis"
            ,"value":["Nephthys", "Hathor", "Horus", "Anubis"]
            ,"Category":"History and Mythology"
            ,"Level":"Hard"
        },
        {
            "ID":33
            ,"question":"This is Austin Powers' middle name."
            ,"Answer":"Danger"
            ,"value":["Ringo", "Swingin' ", "James", "Danger"]
            ,"Category":"Literature and Media"
            ,"Level":"Easy"
        },
        {
            "ID":14
            ,"question":"The Statue of Liberty is green because of"
            ,"Answer":"oxidized copper"
            ,"value":["steel painted green", "oxidized copper", "green stone", "oxidized brass"]
            ,"Category":"Science and Technology"
            ,"Level":"Hard"
        },
        {
            "ID":37
            ,"question":"'Omerta', the title of the last book he completed before his death in 1999, is Sicilian for 'code of silence'"
            ,"Answer":"Mario Puzo"
            ,"value":["Jeffrey Archer", "Kurt Vonnegut", "Truman Capote", "Mario Puzo"]
            ,"Category":"Literature and Media"
            ,"Level":"Medium"
        },
        {
            "ID":23
            ,"question":"In 79 A.D this volcano erupted destroying cities such as Pompeii and Herculaneum"
            ,"Answer":"Mt. Vesuvius"
            ,"value":["Mt. St. Helens", "Mt. Damavand", "Koryakskaya Sopka", "Mt. Vesuvius"]
            ,"Category":"History and Mythology"
            ,"Level":"Medium"
        },
        {
            "ID":22
            ,"question":"This sea route runs along the southern tip of Chile and was first navigated in 1520"
            ,"Answer":"The Strait of Magellan"
            ,"value":["Panama Canal", "English Channel", "The Straits of Gibraltar", "The Strait of Magellan"]
            ,"Category":"History and Mythology"
            ,"Level":"Medium"
        },
        {
            "ID":45
            ,"question":"In what book does Humpty Dumpty first appear?"
            ,"Answer":"Through the Looking Glass"
            ,"value":["Alice in Wonderland", "Through the Looking Glass", "The Wizard of Oz", "Pippi Longstocking"]
            ,"Category":"Literature and Media"
            ,"Level":"Hard"
        },
        {
            "ID":16
            ,"question":"The first conversation recounted in the Bible is in Genesis 3, between these two; it leads to trouble."
            ,"Answer":"Eve and Serpent"
            ,"value":["Eve and Serpent", "Adam and Serpent", "Adam and God", "Adam and Eve"]
            ,"Category":"History and Mythology"
            ,"Level":"Easy"
        }
    ]
    '''

TESTING_TEAM_HAS_SUBJECT = '''
    25,7,18
    26,7,20
    27,7,19
    28,7,17
    37,10,28
    38,10,25
    39,10,26
    40,10,27
    41,11,31
    42,11,30
    43,11,29
    44,11,32
    '''

TESTING_SUBJECT = '''
1,s1,s1,1,\n
2,s2,s2,1,\n
3,s3,s3,1,\n
4,s4,s4,1,\n
5,pogs1.1,pogs1.1,2,\n
6,pogs1.2,pogs1.2,2,\n
7,pogs1.3,pogs1.3,2,\n
8,pogs1.4,pogs1.4,2,\n
9,pogs2.1,pogs2.1,3,\n
10,pogs2.2,pogs2.2,3,\n
11,pogs2.3,pogs2.3,3,\n
12,pogs2.4,pogs2.4,3,\n
13,pogs2.5,pogs2.5,3,\n
14,pogs2.6,pogs2.6,3,\n
15,pogs2.7,pogs2.7,3,\n
16,pogs2.8,pogs2.8,3,\n
17,pogs3.1,pogs3.1,4,\n
18,pogs3.2,pogs3.2,4,\n
19,pogs3.3,pogs3.3,4,\n
20,pogs3.4,pogs3.4,4,\n
21,pogs4.1,pogs4.1,5,\n
22,pogs4.2,pogs4.2,5,\n
23,pogs4.3,pogs4.3,5,\n
24,pogs4.4,pogs4.4,5,\n
25,pogs4.1.1,pogs4.1.1,5,\n
26,pogs4.2.1,pogs4.2.1,5,\n
27,pogs4.3.1,pogs4.3.1,5,\n
28,pogs4.4.1,pogs4.4.1,5,\n
29,pogs5.1,pogs5.1,6,\n
30,pogs5.2,pogs5.2,6,\n
31,pogs5.3,pogs5.3,6,\n
32,pogs5.4,pogs5.4,6,\n
'''

TESTING_LOG_FILE_PATH = '/tmp/event_log.csv'
TESTING_JEOPARDY_FILE_PATH = '/tmp/jeopardy.json'
TESTING_TEAM_HAS_SUBJECT_FILE_PATH = '/tmp/team_has_subject.csv'
TESTING_SUBJECT_FILE_PATH = '/tmp/subject.csv'


class DatasetBuilderTest(unittest.TestCase):

    first_influence_matrix_team7 = np.asarray([
        [.10, .20, .30, .40],
        [.11, .20, .30, .39],
        [.12, .20, .30, .38],
        [.15, .20, .25, .40]
    ])

    second_influence_matrix_team7 = np.asarray([
        [.20, .25, .30, .25],
        [.21, .25, .30, .24],
        [.22, .25, .30, .23],
        [.25, .25, .30, .20]
    ])

    third_influence_matrix_team7 = np.asarray([
        [.30, .25, .20, .25],
        [.31, .25, .20, .24],
        [.32, .25, .20, .23],
        [.35, .25, .20, .20]
    ])

    fourth_influence_matrix_team7 = np.asarray([
        [.40, .30, .20, .10],
        [.41, .30, .20, .9],
        [.42, .30, .20, .8],
        [.45, .30, .20, .5]
    ])

    first_influence_matrix_team10 = np.asarray([
        [.33, .33, .33, .01],
        [.33, .33, .33, .01],
        [.33, .33, .33, .01],
        [.33, .33, .33, .01]
    ])

    second_influence_matrix_team10 = np.asarray([
        [.10, .10, .10, .70],
        [.20, .20, .20, .40],
        [.30, .30, .30, .10],
        [.40, .10, .40, .10]
    ])

    third_influence_matrix_team10 = np.asarray([
        [.20, .20, .30, .30],
        [.50, .10, .20, .20],
        [.75, .15, .05, .05],
        [.35, .35, .15, .15]
    ])

    fourth_influence_matrix_team10 = np.asarray([
        [.85, .05, .05, .05],
        [.35, .35, .15, .15],
        [.25, .25, .25, .25],
        [.40, .30, .20, .10]
    ])

    answers_chosen_team7 = {
        # Easy
        1: {
            17: 'Radio Waves',
            18: 'Radio Waves',
            19: 'Radio Waves',
            20: 'Radio Waves'
        },
        # Hard
        43: {
            17: 'The Tempest',
            18: 'The Tempest',
            19: 'The Tempest',
            20: 'Wrong'
        },
        # Hard
        26: {
            17: 'Taft',
            18: 'Taft',
            19: 'Wrong',
            20: 'Wrong'
        },
        # Hard
        41: {
            17: 'A Goofy Movies',
            18: 'Wrong',
            19: 'Wrong',
            20: 'Wrong',
        },
        # Easy
        4: {
            17: 'Wrong',
            18: 'Wrong',
            19: 'Wrong',
            20: 'Wrong'
        },
        # Hard
        42: {
            17: 'Best Assistant Director',
            18: 'Best Assistant Director',
            19: 'Best Assistant Director',
            20: 'Best Assistant Director'
        },
        # Hard
        29: {
            17: 'Thailand',
            18: 'Thailand',
            19: 'Thailand',
            20: 'Wrong'
        },
        # Easy
        32: {
            17: 'French Revolution',
            18: 'French Revolution',
            19: 'Wrong',
            20: 'Wrong'
        },
        # Hard
        27: {
            17: 'Dwight D Eisenhower',
            18: 'Wrong',
            19: 'Wrong',
            20: 'Wrong'
        },
        # Medium
        6: {
            17: 'Wrong',
            18: 'Wrong',
            19: 'Wrong',
            20: 'Wrong'
        }
    }

    answers_chosen_team10 = {
        # Easy
        1: {
            25: 'Wrong',
            26: 'Wrong',
            27: 'Radio Waves',
            28: 'Radio Waves'
        },
        # Hard
        43: {
            25: 'The Tempest',
            26: 'The Tempest',
            27: 'Wrong',
            28: 'Wrong'
        },
        # Hard
        26: {
            25: 'Wrong',
            26: 'Taft',
            27: 'Taft',
            28: 'Wrong'
        },
        # Hard
        41: {
            25: 'A Goofy Movies',
            26: 'Wrong',
            27: 'Wrong',
            28: 'A Goofy Movies',
        },
        # Easy
        4: {
            25: 'Wrong',
            26: 'Wrong',
            27: 'Wrong',
            28: 'Wrong'
        },
        # Hard
        42: {
            25: 'Best Assistant Director',
            26: 'Wrong',
            27: 'Best Assistant Director',
            28: 'Wrong'
        },
        # Hard
        29: {
            25: 'Wrong',
            26: 'Thailand',
            27: 'Wrong',
            28: 'Thailand'
        },
        # Easy
        32: {
            25: 'French Revolution',
            26: 'French Revolution',
            27: 'Wrong',
            28: 'Wrong'
        },
        # Hard
        27: {
            25: 'Wrong',
            26: 'Wrong',
            27: 'Dwight D Eisenhower',
            28: 'Dwight D Eisenhower'
        },
        # Medium
        6: {
            25: 'Wrong',
            26: 'Wrong',
            27: 'Wrong',
            28: 'Wrong'
        }
    }

    reply_duration_matrix_1_team7 = [
        [1, 2, 3, 4],
        [2, 3, 4, 1],
        [3, 4, 1, 2],
        [1, 2, 3, 4]
    ]

    @classmethod
    def setUpClass(cls):

        # Setup for Team 7
        tlp_team7 = pogs_lib.TeamLogProcessor(
            7, '/Users/koasato/Documents/koa/College/UCSB/2019-2020/Research/Jeopardy/', pogs_lib.JeopardyInfoOptions())
        # Reorganize the members array
        tlp_team7.members = [17, 18, 19, 20]

        # Set the memeber influences for the first four influences
        tlp_team7.member_influences = [
            cls.first_influence_matrix_team7,
            cls.second_influence_matrix_team7,
            cls.third_influence_matrix_team7,
            cls.fourth_influence_matrix_team7
        ]

        # Change answers for easier checking
        tlp_team7.individual_answers_chosen[1] = cls.answers_chosen_team7[1]
        tlp_team7.individual_answers_chosen[43] = cls.answers_chosen_team7[43]
        tlp_team7.individual_answers_chosen[26] = cls.answers_chosen_team7[26]
        tlp_team7.individual_answers_chosen[41] = cls.answers_chosen_team7[41]
        tlp_team7.individual_answers_chosen[4] = cls.answers_chosen_team7[4]
        tlp_team7.individual_answers_chosen[42] = cls.answers_chosen_team7[42]
        tlp_team7.individual_answers_chosen[29] = cls.answers_chosen_team7[29]
        tlp_team7.individual_answers_chosen[32] = cls.answers_chosen_team7[32]
        tlp_team7.individual_answers_chosen[27] = cls.answers_chosen_team7[27]
        tlp_team7.individual_answers_chosen[6] = cls.answers_chosen_team7[6]

        # Setup for Team 10
        tlp_team10 = pogs_lib.TeamLogProcessor(
            10, '/Users/koasato/Documents/koa/College/UCSB/2019-2020/Research/Jeopardy/', pogs_lib.JeopardyInfoOptions())

        # Reorganize the members array
        tlp_team10.members = [25, 26, 27, 28]

        # Set the member influences for the first four influences
        tlp_team10.member_influences = [
            cls.first_influence_matrix_team10,
            cls.second_influence_matrix_team10,
            cls.third_influence_matrix_team10,
            cls.fourth_influence_matrix_team10
        ]

        # Change answers for easier checking
        tlp_team10.individual_answers_chosen[1] = cls.answers_chosen_team10[1]
        tlp_team10.individual_answers_chosen[43] = cls.answers_chosen_team10[43]
        tlp_team10.individual_answers_chosen[26] = cls.answers_chosen_team10[26]
        tlp_team10.individual_answers_chosen[41] = cls.answers_chosen_team10[41]
        tlp_team10.individual_answers_chosen[4] = cls.answers_chosen_team10[4]
        tlp_team10.individual_answers_chosen[42] = cls.answers_chosen_team10[42]
        tlp_team10.individual_answers_chosen[29] = cls.answers_chosen_team10[29]
        tlp_team10.individual_answers_chosen[32] = cls.answers_chosen_team10[32]
        tlp_team10.individual_answers_chosen[27] = cls.answers_chosen_team10[27]
        tlp_team10.individual_answers_chosen[6] = cls.answers_chosen_team10[6]

        tlps = {}
        tlps[7] = tlp_team7
        tlps[10] = tlp_team10
        # Call dataset builder with
        cls.loader = data_builder.DatasetBuilder(tlps)

        # Change content embeddings for team 7
        cls.loader.contents_embeddings[7][0] = [0, 1, 2, 3]
        cls.loader.contents_embeddings[7][1] = [1, 1, 2, 3]
        cls.loader.contents_embeddings[7][2] = [2, 1, 2, 3]
        cls.loader.contents_embeddings[7][3] = [3, 1, 2, 3]

        # Change content embeddings for team 7
        cls.loader.contents_embeddings[10][0] = [4, 5, 6, 7]
        cls.loader.contents_embeddings[10][1] = [8, 9, 10, 11]
        cls.loader.contents_embeddings[10][2] = [-1, -2, -3, -4]
        cls.loader.contents_embeddings[10][3] = [80, 80, 100, 100]

        cls.loader.create_dataset()

        cls.data = cls.loader.supervised_data

    def test_individual_performance_team7(self):
        correct_individual_performance_round_1 = np.asarray(
            [8/10, 6/10, 4/10, 2/10])

        np.testing.assert_array_almost_equal(correct_individual_performance_round_1,
                                             self.data['X'][0]['individual_performance'])

    def test_individual_performance_team10(self):
        correct_individual_performance_round_1 = np.asarray(
            [4/10, 4/10, 4/10, 4/10])

        np.testing.assert_array_almost_equal(correct_individual_performance_round_1,
                                             self.data['X'][3]['individual_performance'])

    def test_individual_performance_hardness_weighted_team7(self):
        correct_individual_performance_weighted_round_1 = np.asarray(
            [20/23, 14/23, 10/23, 4/23]
        )

        np.testing.assert_array_almost_equal(correct_individual_performance_weighted_round_1,
                                             self.data['X'][0]['individual_performance_hardness_weighted'])

    def test_individual_performance_hardness_weighted_team10(self):
        correct_individual_performance_weighted_round_1 = np.asarray(
            [10/23, 10/23, 10/23, 10/23]
        )

        np.testing.assert_array_almost_equal(correct_individual_performance_weighted_round_1,
                                             self.data['X'][3]['individual_performance_hardness_weighted'])

    def test_content_embedding_matrix_for_team7(self):
        # We ignore embeddings_1 since it's the first for that team
        correct_content_embeddings_1 = np.asarray([1, 1, 2, 3])
        correct_content_embeddings_2 = np.asarray([2, 1, 2, 3])
        correct_content_embeddings_3 = np.asarray([3, 1, 2, 3])
        correct_content_embeddings_4 = np.asarray([4, 1, 2, 3])

        # Embeddings for team 7
        np.testing.assert_array_almost_equal(correct_content_embeddings_1,
                                             self.data['X'][0]['content_embedding_matrix'])
        np.testing.assert_array_almost_equal(correct_content_embeddings_2,
                                             self.data['X'][1]['content_embedding_matrix'])
        np.testing.assert_array_almost_equal(correct_content_embeddings_3,
                                             self.data['X'][2]['content_embedding_matrix'])

    def test_content_embedding_matrix_for_team10(self):
        # We ignore embeddings_5 since it's the first for that team
        correct_content_embeddings_5 = np.asarray([4, 5, 6, 7])
        correct_content_embeddings_6 = np.asarray([8, 9, 10, 11])
        correct_content_embeddings_7 = np.asarray([-1, -2, -3, -4])
        correct_content_embeddings_8 = np.asarray([80, 80, 100, 100])

        # Embeddings for team 10
        np.testing.assert_array_almost_equal(correct_content_embeddings_6,
                                             self.data['X'][3]['content_embedding_matrix'])
        np.testing.assert_array_almost_equal(correct_content_embeddings_7,
                                             self.data['X'][4]['content_embedding_matrix'])
        np.testing.assert_array_almost_equal(correct_content_embeddings_8,
                                             self.data['X'][5]['content_embedding_matrix'])

    def test_first_influence_matrix_team7(self):
        correct_first_influence_matrix_team7 = self.first_influence_matrix_team7

        np.testing.assert_array_almost_equal(correct_first_influence_matrix_team7,
                                             self.data['X'][0]['first_influence_matrix'])
        np.testing.assert_array_almost_equal(correct_first_influence_matrix_team7,
                                             self.data['X'][1]['first_influence_matrix'])
        np.testing.assert_array_almost_equal(correct_first_influence_matrix_team7,
                                             self.data['X'][2]['first_influence_matrix'])

    def test_first_influence_matrix_team10(self):
        correct_first_influence_matrix_team10 = self.first_influence_matrix_team10

        np.testing.assert_array_almost_equal(correct_first_influence_matrix_team10,
                                             self.data['X'][3]['first_influence_matrix'])
        np.testing.assert_array_almost_equal(correct_first_influence_matrix_team10,
                                             self.data['X'][4]['first_influence_matrix'])
        np.testing.assert_array_almost_equal(correct_first_influence_matrix_team10,
                                             self.data['X'][5]['first_influence_matrix'])

    def test_previous_influence_matrix_team7(self):
        correct_previous_influence_matrix_1 = self.first_influence_matrix_team7

        correct_previous_influence_matrix_2 = self.second_influence_matrix_team7

        correct_previous_influence_matrix_3 = self.third_influence_matrix_team7

        np.testing.assert_array_almost_equal(correct_previous_influence_matrix_1,
                                             self.data['X'][0]['previous_influence_matrix'])
        np.testing.assert_array_almost_equal(correct_previous_influence_matrix_2,
                                             self.data['X'][1]['previous_influence_matrix'])
        np.testing.assert_array_almost_equal(correct_previous_influence_matrix_3,
                                             self.data['X'][2]['previous_influence_matrix'])

    def test_previous_influence_matrix_team10(self):
        correct_previous_influence_matrix_1 = self.first_influence_matrix_team10

        correct_previous_influence_matrix_2 = self.second_influence_matrix_team10

        correct_previous_influence_matrix_3 = self.third_influence_matrix_team10

        np.testing.assert_array_almost_equal(correct_previous_influence_matrix_1,
                                             self.data['X'][3]['previous_influence_matrix'])
        np.testing.assert_array_almost_equal(correct_previous_influence_matrix_2,
                                             self.data['X'][4]['previous_influence_matrix'])
        np.testing.assert_array_almost_equal(correct_previous_influence_matrix_3,
                                             self.data['X'][5]['previous_influence_matrix'])

    def test_average_of_previous_influence_matrices_team7(self):
        correct_average_of_previous_influence_matrices_round_1 = \
            self.first_influence_matrix_team7 / 1

        correct_average_of_previous_influence_matrices_round_2 = \
            (self.first_influence_matrix_team7 +
             self.second_influence_matrix_team7) / 2

        correct_average_of_previous_influence_matrices_round_3 = \
            (self.first_influence_matrix_team7 + self.second_influence_matrix_team7 +
             self.third_influence_matrix_team7) / 3

        np.testing.assert_array_almost_equal(correct_average_of_previous_influence_matrices_round_1,
                                             self.data['X'][0]['average_of_previous_influence_matrices'])
        np.testing.assert_array_almost_equal(correct_average_of_previous_influence_matrices_round_2,
                                             self.data['X'][1]['average_of_previous_influence_matrices'])
        np.testing.assert_array_almost_equal(correct_average_of_previous_influence_matrices_round_3,
                                             self.data['X'][2]['average_of_previous_influence_matrices'])

    def test_average_of_previous_influence_matrices_team10(self):
        correct_average_of_previous_influence_matrices_round_1 = \
            self.first_influence_matrix_team10 / 1

        correct_average_of_previous_influence_matrices_round_2 = \
            (self.first_influence_matrix_team10 +
             self.second_influence_matrix_team10) / 2

        correct_average_of_previous_influence_matrices_round_3 = \
            (self.first_influence_matrix_team10 + self.second_influence_matrix_team10 +
             self.third_influence_matrix_team10) / 3

        np.testing.assert_array_almost_equal(correct_average_of_previous_influence_matrices_round_1,
                                             self.data['X'][3]['average_of_previous_influence_matrices'])
        np.testing.assert_array_almost_equal(correct_average_of_previous_influence_matrices_round_2,
                                             self.data['X'][4]['average_of_previous_influence_matrices'])
        np.testing.assert_array_almost_equal(correct_average_of_previous_influence_matrices_round_3,
                                             self.data['X'][5]['average_of_previous_influence_matrices'])

    def test_reply_duration_matrix(self):

        print(self.data['X'][0]['reply_duration'])
