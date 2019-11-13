# Omid55
# Test module for pogs jeopardy log library.
from __future__ import division, print_function, absolute_import, unicode_literals

import pandas as pd
import numpy as np
import mock
import os
import unittest
from parameterized import parameterized
from numpy import testing as np_testing

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

INFLUENCE_TESTING_LOG = '''
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
'''
TESTING_LOG_FILE_PATH = '/tmp/event_log.csv'
TESTING_JEOPARDY_FILE_PATH = '/tmp/jeopardy.json'
TESTING_TEAM_HAS_SUBJECT_FILE_PATH = '/tmp/team_has_subject.csv'
TESTING_SUBJECT_PATH = '/tmp/subject.csv'


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

    def test_load_this_team_event_logs_loads_team_logs(self):
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

    def test_load_messages_has_loaded_log(self):
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
        self.assertEqual(individual_responses[41] ==
            {17: 'Chicken Little', 18: 'Chicken Little',
            19: 'Chicken Little', 20: 'Chicken Little'}, True)
        self.assertEqual(group_responses[41] ==
            {20: 'Chicken Little', 19: 'Chicken Little', 18:
            'Chicken Little', 17: 'Chicken Little'}, True)
        self.assertEqual(individual_responses[42] ==
            {17: 'Best Sound Mixing', 18: 'Best Original Score',
            19: 'Best Sound Mixing', 20: 'Best Sound Mixing'}, True)
        self.assertEqual(group_responses[42] ==
            {19: 'Best Sound Mixing', 17: 'Best Sound Mixing',
            20: 'Best Sound Mixing', 18: 'Best Sound Mixing'}, True)


# =========================================================================
# ======================== _load_machine_usage_info =======================
# =========================================================================
class TeamLogProcessorMachineUsageTest(unittest.TestCase):

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
            cls.loader._load_machine_usage_info()

    @classmethod
    def tearDownClass(cls):
        os.remove(TESTING_LOG_FILE_PATH)
        os.remove(TESTING_TEAM_HAS_SUBJECT_FILE_PATH)
        os.remove(TESTING_JEOPARDY_FILE_PATH)

    def test_size_of_machine_usage_info(self):
        self.assertEqual(len(self.loader.machine_usage_info), 6)

    def test_keys_of_machine_usage_info(self):
        keys = [1, 4, 26, 41, 42, 43]
        self.assertEqual(all(key in keys for key in self.loader.machine_usage_info.keys()), True)
        self.assertEqual(all(key in self.loader.machine_usage_info.keys() for key in keys), True)

    def test_machine_was_not_used_for_question_26(self):
        self.assertEqual(self.loader.machine_usage_info[26].used, False)
        self.assertEqual(self.loader.machine_usage_info[26].user, -1)
        self.assertEqual(self.loader.machine_usage_info[26].answer_given, "")
        self.assertEqual(self.loader.machine_usage_info[26].probability, -1)

    def test_machine_info_for_question_42(self):
        self.assertEqual(self.loader.machine_usage_info[42].used, True)
        self.assertEqual(self.loader.machine_usage_info[42].user, 20)
        self.assertEqual(self.loader.machine_usage_info[42].answer_given, "Best Sound Mixing")
        self.assertEqual(self.loader.machine_usage_info[42].probability, 0.6)


# =========================================================================
# ======================== _load_ratings ==================================
# =========================================================================
class TeamLogProcessorLoadRatingsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(TESTING_LOG_FILE_PATH, 'w') as f:
            f.writelines(INFLUENCE_TESTING_LOG)
        with open(TESTING_TEAM_HAS_SUBJECT_FILE_PATH, 'w') as f:
            f.writelines(TESTING_TEAM_HAS_SUBJECT)
        with open(TESTING_JEOPARDY_FILE_PATH, 'w') as f:
            f.writelines(TESTING_JEOPARDY_JSON)
        with open(TESTING_SUBJECT_PATH, 'w') as f:
            f.writelines(TESTING_SUBJECT)
        with mock.patch.object(lib.TeamLogProcessor, '_load_all_files'):
            cls.loader = lib.TeamLogProcessor(
                team_id=1, logs_directory_path='tmp')
            cls.loader.logs_directory_path='/tmp/'
            cls.loader._load_this_team_event_logs(
                logs_file_path=TESTING_LOG_FILE_PATH,
                team_has_subject_file_path=TESTING_TEAM_HAS_SUBJECT_FILE_PATH)
            cls.loader._preload_data(directory='/tmp/')
            cls.loader._define_team_member_order(directory='/tmp/')
            cls.loader._load_ratings()

    @classmethod
    def tearDownClass(cls):
        os.remove(TESTING_LOG_FILE_PATH)
        os.remove(TESTING_TEAM_HAS_SUBJECT_FILE_PATH)

    def test_team_array(self):
        correct_team_array = ['pogs3.2', 'pogs3.4', 'pogs3.3', 'pogs3.1']
        self.assertEqual(correct_team_array, self.loader.team_array)

    def test_agent_ratings(self):
        correct_agent_ratings_0 = [
            [0.0, 0.0, 0.0, 0.0], [0.3, 0.1, 0.3, 0.3],
            [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        correct_agent_ratings_1 = [
            [2.0, 2.0, 4.0, 4.0], [4.0, 4.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
        correct_agent_ratings_2 = [
            [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
        correct_agent_ratings_3 = [
            [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]

        self.assertEqual(correct_agent_ratings_0, self.loader.agent_ratings[0])
        self.assertEqual(correct_agent_ratings_1, self.loader.agent_ratings[1])
        self.assertEqual(correct_agent_ratings_2, self.loader.agent_ratings[2])
        self.assertEqual(correct_agent_ratings_3, self.loader.agent_ratings[3])

    def test_member_influences(self):
        correct_member_influences_0 = [
            [25, 25, 25, 25], [40, 0, 40, 20], [0, 100, 0, 0], [1, 1, 1, 97]]
        correct_member_influences_1 = [
            [10, 20, 30, 40], [40, 30, 20, 10], [5, 20, 5, 70], [1, 1, 1, 97]]
        correct_member_influences_2 = [
            [25, 25, 25, 25], [25, 25, 25, 25], [10, 20, 10, 60], [1, 1, 1, 97]]
        correct_member_influences_3 = [
            [25, 25, 25, 25], [25, 25, 25, 25], [20, 30, 20, 30], [0, 2, 2, 96]]

        self.assertEqual(correct_member_influences_0, self.loader.member_influences[0])
        self.assertEqual(correct_member_influences_1, self.loader.member_influences[1])
        self.assertEqual(correct_member_influences_2, self.loader.member_influences[2])
        self.assertEqual(correct_member_influences_3, self.loader.member_influences[3])

    def test_agent_ratings_from_data(self):
        correct_agent_ratings_from_data_0 = [
            [False, False, False, False],[True, True, True, True],
            [True, True, True, True], [True, True, True, True]]
        correct_agent_ratings_from_data_1 = [
            [True, True, True, True], [True, True, True, True],
            [True, True, True, True], [True, True, True, True]]
        correct_agent_ratings_from_data_2 = [
            [False, False, False, False], [False, False, False, False],
            [True, True, True, True], [True, True, True, True]]
        correct_agent_ratings_from_data_3 = [
            [False, False, True, True], [False, False, False, False],
            [True, True, True, True], [True, True, True, True]]

        self.assertEqual(correct_agent_ratings_from_data_0, self.loader.agent_ratings_from_data[0])
        self.assertEqual(correct_agent_ratings_from_data_1, self.loader.agent_ratings_from_data[1])
        self.assertEqual(correct_agent_ratings_from_data_2, self.loader.agent_ratings_from_data[2])
        self.assertEqual(correct_agent_ratings_from_data_3, self.loader.agent_ratings_from_data[3])

    def test_member_influences_from_data(self):
        correct_member_influences_from_data_0 = [
            [False, False, False, False],[True, True, True, True],
            [True, True, True, True], [True, True, True, True]]
        correct_member_influences_from_data_1 = [
            [True, True, True, True], [True, True, True, True],
            [True, True, True, True], [True, True, True, True]]
        correct_member_influences_from_data_2 = [
            [False, False, False, False], [False, False, False, False],
            [True, True, True, True], [True, True, True, True]]
        correct_member_influences_from_data_3 = [
            [False, False, False, False], [False, False, False, False],
            [True, True, True, True], [True, True, True, True]]

        self.assertEqual(correct_member_influences_from_data_0, self.loader.member_influences_from_data[0])
        self.assertEqual(correct_member_influences_from_data_1, self.loader.member_influences_from_data[1])
        self.assertEqual(correct_member_influences_from_data_2, self.loader.member_influences_from_data[2])
        self.assertEqual(correct_member_influences_from_data_3, self.loader.member_influences_from_data[3])

# =========================================================================
# ======================== _load_accumulated_score ========================
# =========================================================================
class TeamLogProcessorScoreTest(unittest.TestCase):

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
            cls.loader._load_machine_usage_info()
            cls.loader._load_accumulated_score()

    @classmethod
    def tearDownClass(cls):
        os.remove(TESTING_LOG_FILE_PATH)
        os.remove(TESTING_TEAM_HAS_SUBJECT_FILE_PATH)

    def test_size_of_score(self):
        self.assertEqual(len(self.loader.score), 6)

    def test_score_has_correct_keys(self):
        keys = [1, 4, 26, 41, 42, 43]
        self.assertEqual(all(key in keys for key in self.loader.score.keys()), True)
        self.assertEqual(all(key in self.loader.score.keys() for key in keys), True)

    def test_score_has_correct_values(self):
        self.assertEqual(self.loader.score[1] == 4, True)
        self.assertEqual(self.loader.score[43] == -1, True)
        self.assertEqual(self.loader.score[26] == -1, True)
        self.assertEqual(self.loader.score[41] == -1, True)
        self.assertEqual(self.loader.score[4] == 4, True)
        self.assertEqual(self.loader.score[42] == -2, True)

    # Accumulated score has an extra index at 0 as the starting point
    def test_size_of_accumulated_score(self):
        self.assertEqual(len(self.loader.accumulated_score), 7)

    def test_accumulated_score_has_correct_values(self):
        self.assertEqual(self.loader.accumulated_score[1] == 4, True)
        self.assertEqual(self.loader.accumulated_score[2] == 3, True)
        self.assertEqual(self.loader.accumulated_score[3] == 2, True)
        self.assertEqual(self.loader.accumulated_score[4] == 1, True)
        self.assertEqual(self.loader.accumulated_score[5] == 5, True)
        self.assertEqual(self.loader.accumulated_score[6] == 3, True)

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

#     def test_load_influence_matrices_has_loaded_log(self):
#         # self.loader.member_influences
#         pass
