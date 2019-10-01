import pandas as pd 
import numpy as np
import fileinput
import json
from scipy.stats import beta
import matplotlib.pyplot as plt
import re
import networkx as nx 
import math 
from scipy.stats import wilcoxon
from statistics import mean 
from scipy.stats import pearsonr
# from cpt_valuation import evaluateProspectVals


class HumanDecisionModels:
    def __init__(self,teamId,directory):
        
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
        newEventContent = pd.DataFrame(index=np.arange(0, len(eventLogTaskAttribute)), columns=("id","stringValue", "questionNumber","questionScore","attributeName"))
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
        
        for i in range(0, self.numQuestions):
            self.correctAnswers.append(d[int(float(self.questionNumbers[i]))-1]['Answer'])
            self.options.append(d[int(float(self.questionNumbers[i]))-1]['value'])
        
        allIndividualResponses = eventLogWithAllData[eventLogWithAllData['extra_data'] == "IndividualResponse"]
        self.lastIndividualResponsesbyQNo = allIndividualResponses.groupby(['sender', 'questionNumber'], as_index=False, sort=False).last()
        
        # Compute the group answer of the team per question
        submissions = eventLogWithAllData[(eventLogWithAllData['extra_data'] == "IndividualResponse") | (eventLogWithAllData['extra_data'] == "GroupRadioResponse") ]
        individualAnswersPerQuestion = submissions.groupby(["questionNumber","sender_subject_id"], as_index=False, sort=False).tail(1)
        
        self.groupSubmission = pd.DataFrame(index=np.arange(0, len(self.questionNumbers)), columns=("questionNumber","groupAnswer"))
        for i in range(0, self.numQuestions):
            ans = ""
            consensusReached = True
            for j in range(0,len(individualAnswersPerQuestion)):                    
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
        for i in range(self.numQuestions):
            if int(float(self.questionNumbers[i])) in self.machineAskedQuestions:
                indxM = self.machineAskedQuestions.index(int(float(self.questionNumbers[i])))            
                k = self.teamArray.index(self.machineAsked['sender'].iloc[indxM])
                self.machineUsed[i][int(k)] = True
        
        self.teamScore = list()           
        self.computeTeamScore()
                
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
        
        # Hyperparameters for expected performance (Humans and Agents) - TODO
        self.alphas = [1,1,1,1,1,1,1,1]
        self.betas = np.ones(8, dtype = int)

        #vector c
        self.centralities = [[] for _ in range(self.numQuestions)]

        self.actionTaken = list()
        self.computeActionTaken()
      
    def computeTeamScore(self):
        self.teamScore.append(0)
        for i in range(0,self.numQuestions):
            if self.groupSubmission.groupAnswer[i]!=self.correctAnswers[i]:
                self.teamScore[i]+=self.z
            else:
                self.teamScore[i]+=self.c    
            if len(np.where(self.machineUsed[i] == True)[0])!=0:
                self.teamScore[i]+=self.e
            self.teamScore.append(self.teamScore[i])
        self.teamScore = self.teamScore[:-1]    
    
    def updateAlphaBeta(self, i, valueSubmitted, correctAnswer):
        if (valueSubmitted == correctAnswer):
            self.alphas[i]+=1
        else:
            self.betas[i]+=1 
            
    def naiveProbability(self, questionNumber, idx):
        expectedPerformance = list()
        individualResponse = list()
        probabilities = list()
        human_accuracy = list()

        machine_accuracy = [None for _ in range(self.numAgents)]
        group_accuracy = 0
        
        #Save human expected performance based
        for i in range(0,self.teamSize):
            expectedPerformance.append(beta.mean(self.alphas[i],self.betas[i]))
            individualResponse.append(self.lastIndividualResponsesbyQNo[(self.lastIndividualResponsesbyQNo["questionNumber"] == questionNumber) & (self.lastIndividualResponsesbyQNo["sender"] == self.teamMember.iloc[i])]["stringValue"].any())
            self.updateAlphaBeta(i,self.lastIndividualResponsesbyQNo[(self.lastIndividualResponsesbyQNo["questionNumber"] == questionNumber) & (self.lastIndividualResponsesbyQNo["sender"] == self.teamMember.iloc[i])]["stringValue"].any(),self.correctAnswers[idx])

            ans = self.lastIndividualResponsesbyQNo[(self.lastIndividualResponsesbyQNo["questionNumber"] == questionNumber) & (self.lastIndividualResponsesbyQNo["sender"] == self.teamMember.iloc[i])]["stringValue"].any()
            if ans == self.correctAnswers[idx]:
                human_accuracy.append(1)
            else:
                human_accuracy.append(0)
                
        if (self.groupSubmission["groupAnswer"].iloc[idx] == self.correctAnswers[idx]):
            group_accuracy = 1            
            
        indxQ = -1
        anyMachineAsked = False
        if(int(float(questionNumber)) in self.machineAskedQuestions):
            indxQ = self.machineAskedQuestions.index(int(float(questionNumber)))        
            sender = self.machineAsked['sender'].iloc[indxQ]
            k = self.teamArray.index(sender)
            anyMachineAsked = True
        
        # Add expected Performance for Agents
        for i in range(self.teamSize, self.teamSize+self.numAgents):
            expectedPerformance.append(beta.mean(self.alphas[i],self.betas[i]))
            # update alpha beta for that machine

        #Update machine accuracy 
        if(anyMachineAsked):
            self.updateAlphaBeta(self.getAgentForHuman(k), self.machineAsked['event_content'].iloc[indxQ].split("||")[0].split(":")[2].replace('"', ''), self.correctAnswers[idx])
            self.machineUseCount[k]+=1
            machineAnswer = self.machineAsked['event_content'].iloc[indxQ].split("||")[0].split(":")[2].replace('"', '').split("_")[0]

            if self.firstMachineUsage[k] == -1:
                self.firstMachineUsage[k] = idx
            machine_accuracy[k] = 1

        
        # Conditional Probability
        # Do a bayes update
        denominator = 0
        numerator = [1. for _ in range(len(self.options[idx]))]
        prob_class = 0.25
        prob_resp = 0
        prob_class_responses = [None for _ in range(len(self.options[idx]))]
        prob_resp_given_class = [None for _ in range(len(self.options[idx]))]


        for opt_num in range(0,len(self.options[idx])):
            prob_resp = 0
            numerator = prob_class
            for person_num in range(0,self.teamSize):
                if individualResponse[person_num] == self.options[idx][opt_num]:
                    numerator *= expectedPerformance[person_num]
                else:
                    numerator *= (1 - expectedPerformance[person_num])/3
                prob_resp += numerator

            prob_resp_given_class[opt_num] = numerator
        prob_class_responses = [(prob_resp_given_class[i]/sum(prob_resp_given_class)) for i in range(0,len(prob_resp_given_class))]
        
        #ANSIs this updating agent probabilities?
        for i in range(self.teamSize):
            probabilities.append(expectedPerformance[self.teamSize+i])    

        #8 probability values returned
        # first set is for options (sums to 1)
	
        assert(sum(prob_class_responses) > 0.999 and sum(prob_class_responses) < 1.001)
        #second set is for machines
        prob_all_class_responses = prob_class_responses + [expectedPerformance[self.getAgentForHuman(k)] for k in range(self.teamSize)]

        return prob_all_class_responses,human_accuracy,group_accuracy,machine_accuracy 
    
    def updateCentrality(self, influenceMatrixIndex):
        #Compute Eigen Vector Centrality for Humans
        graph = nx.DiGraph()
        for i in range(0,self.teamSize):
            for j in range(0,self.teamSize):                
                graph.add_edge(i,j,weight=self.memberInfluences[influenceMatrixIndex][i][j]/100)

        human_centralities = nx.eigenvector_centrality(graph, weight="weight")

        #Compute expected performance for machines

        """
        for i in range(0,self.teamSize):
            numerator = 0
            denominator = 0
            for j in range(0,self.teamSize):
                numerator+= self.centralities[j] * self.agentRatings[influenceMatrixIndex][j][i]
                denominator+= self.centralities[j]    
            self.centralities.update({self.teamSize+i:numerator/denominator})   

        """
        #Check that we have the correct total influence 
        for i in range(self.teamSize):
            assert(sum(self.memberInfluences[influenceMatrixIndex][i][j] for j in range(self.numAgents)) == 100)

        #Make a probability

        agent_weighted_centrality_perf = [None for _ in range(self.numAgents)]
        for i in range(self.numAgents):
            agent_weighted_centrality_perf[i] = sum([self.memberInfluences[influenceMatrixIndex][i][j]/100. for j in range(self.numAgents)])

        centralities_as_list = [value for value in human_centralities.values()]
        for question_num in range(self.influenceMatrixIndex*5 ,(self.influenceMatrixIndex+1)*5):
            self.centralities[question_num] = centralities_as_list + agent_weighted_centrality_perf

        #Move to next influence matrix
        self.influenceMatrixIndex+=1        
        
    def calculatePerformanceProbability(self, questionNumber, idx):

        probabilities = list()
        probabilities = [0 for _ in range(self.teamSize + self.numAgents)]
            

        for i in range(0,self.teamSize):
            individulResponse = self.lastIndividualResponsesbyQNo[(self.lastIndividualResponsesbyQNo["questionNumber"] == questionNumber) & (self.lastIndividualResponsesbyQNo["sender"] == self.teamMember.iloc[i])]["stringValue"].any()
            index = self.options[idx].index(individulResponse)   
            probabilities[index] += self.centralities[idx][i]
        
        # Normalize the probabilties
        totalProbability = sum(probabilities)
        probabilities[:] = [x / totalProbability for x in probabilities] 
               
        # Add expected Performance for Agents
        for i in range(0, self.numAgents):                       
            #which agents should have a centrality of 1?
            if self.centralities[idx][self.getAgentForHuman(i)] == 1:
                probabilities[self.getAgentForHuman(i)] = self.centralities[idx][self.getAgentForHuman(i)]
            #which agents should have a positive centrality 
            elif self.centralities[idx][i+self.teamSize] >= 0:
                probabilities[self.getAgentForHuman(i)] = self.centralities[idx][self.getAgentForHuman(i)]     
            else:
                assert(False) # no negative centralities allowed

        return probabilities    
    
    def calculateModelAccuracy(self,perQuestionRewards,probabilities,idx):
        highestRewardOption = max(perQuestionRewards[0:4])
        highestRewardAgent = max(perQuestionRewards[4:8])
        modelAccuracy = 0
        count = 0
        
        if highestRewardOption >= highestRewardAgent:
            for i in range(0,self.teamSize):
                if highestRewardOption == perQuestionRewards[i] and self.options[idx][i]==self.correctAnswers[idx]:
                    count+=1
                    modelAccuracy = 1
            modelAccuracy = modelAccuracy * count / (perQuestionRewards[0:4].count(highestRewardOption))
        else:
            for i in range(self.teamSize,self.teamSize*2):
                if highestRewardAgent == perQuestionRewards[i]:
                    modelAccuracy += probabilities[i] * (perQuestionRewards[4:8].count(highestRewardAgent))                                   
        return modelAccuracy
    
    # Expected rewards for (all options + all agents)
    def calculateExpectedReward(self, probabilities):
        perQuestionRewards = list()            
        for j in range(0,self.teamSize):
            perQuestionRewards.append(self.c*probabilities[j] + (self.z)*(1-probabilities[j])) 

        for j in range(0,self.teamSize):
            perQuestionRewards.append((self.c+self.e)*probabilities[self.getAgentForHuman(j)] + (self.z+self.e)*(1-probabilities[self.getAgentForHuman(j)]))

        return perQuestionRewards
    
    def calculateRewards(self):
        rewardsNB1 = list()
        probabilitiesNB1 = list()
        
        # Compute Reward for NB1
        for i in range(0,self.numQuestions):                    
            probabilities,accuracy, group_accuracy, machine_accuracy = self.naiveProbability(self.questionNumbers[i],i)
        
        for i in range(0,self.numQuestions):                      
            all_probabilities, human_accuracy, group_accuracy, machine_accuracy = self.naiveProbability(self.questionNumbers[i],i)
            probabilitiesNB1.append(all_probabilities)
            rewardsNB1.append(self.calculateExpectedReward(all_probabilities))

        #Compute Reward for CENT1 model

        rewardsCENT1 = list()
        probabilitiesCENT1 = list()

        for i in range(0,self.numCentralityReports):            
            self.updateCentrality(self.influenceMatrixIndex)             

        for i in range(0,self.numQuestions):            
            probabilities = self.calculatePerformanceProbability(self.questionNumbers[i],i)
            probabilitiesCENT1.append(probabilities)
            rewardsCENT1.append(self.calculateExpectedReward(probabilities))

        return rewardsNB1,rewardsCENT1, probabilitiesNB1,probabilitiesCENT1               
    
    #--Deprecated--
    def computePTaccuracy(self, pi):
        PTrewards = list()
        for i in range(0,len(pi)):
            PTrewards.append(model.calculateExpectedReward(pi[i]))
        accuracy = list()
        for i in range(0,len(pi)):                             
            if i==0:
                accuracy.append(self. calculateModelAccuracy(PTrewards[i],pi[i],(i+self.trainingSetSize))/(i+1))
            else:
                accuracy.append((self.calculateModelAccuracy(PTrewards[i],pi[i],(i+self.trainingSetSize)) + (i*accuracy[i-1]))/(i+1)) 
        return PTrewards, accuracy     
    
    def softmax(self, vec, index): 
        return (np.exp(vec) / np.sum(np.exp(vec), axis=0))[index]
    
        # Called in loss function --Deprecated--
    def newValues(self,values):
        least = min(values)
        values[:] = [i-least for i in values]
        values[:] = [i/sum(values) for i in values]
        return values
   
    def computeActionTaken(self):        
        for i in range(0,self.numQuestions):
            if self.groupSubmission.groupAnswer[i] == "Consensus Not Reached":
                self.actionTaken.append(-1)
            elif int(float(self.questionNumbers[i])) in self.machineAskedQuestions:
                self.actionTaken.append(self.teamSize + np.where(self.machineUsed[i] == True)[0][0])
            else:
                self.actionTaken.append(self.options[i].index(self.groupSubmission.groupAnswer[i]))
                
#     Computes V1 to V8 for a given question --Deprecated--
    def computeCPT(self,alpha,gamma,probabilities):
        values = list()
        for i in range(0,2*self.teamSize):
            if i<4:
                values.append((math.pow(self.c, alpha) * math.exp(-math.pow(math.log(1/probabilities[i]), gamma)))-(math.pow(math.fabs(self.z), alpha) * math.exp(-math.pow(math.log(1/(1-probabilities[i])), gamma))))
            else:
                values.append((math.pow(self.c+self.z, alpha) * math.exp(-math.pow(math.log(1/probabilities[i]), gamma)))-(math.pow(math.fabs(self.z + self.e), alpha) * math.exp(-math.pow(math.log(1/(1-probabilities[i])), gamma))))
        return values
    
    #--Deprecated--
    def bestAlternative(self,values,action):
        highest = max(values)
        if highest!=action:
            return highest
        temp = list(filter(lambda a: a != highest, values))
        if len(temp)==0:
            return -100
        return max(temp)
    
#     Compute P_I for CPT models --Deprecated--
    def computePI(self, values, actionTaken,lossType): 
        z = self.bestAlternative(values,values[actionTaken])
        if (z==-100):
            if lossType=="logit":
                return 0.25 
            else:
                return 0
        z = values[actionTaken]-z        
        if lossType=="softmax":
            return z
        return 1/(1+math.exp(-z))    
    

    #action in 0,...,numAgents
    def computeAgentLoss(self,params,probabilities,chosen_action,lossType,modelName):
        current_models = ["nb","nb-pt","cent","cent-pt"]
        if (modelName not in current_models):
            assert(False)

        prospects= []
        for probability in probabilities:
            prospectSuccess = self.c +self.e, probability
            prospectFailure = self.z +self.e, 1-probability
            prospects.append((prospectSuccess,prospectFailure))
        pass
        # cpt_vals = evaluateProspectVals(params,prospects)
        # arg = self.softmax(cpt_vals,chosen_action)

        # return -1.*math.log(arg)

    #action in 0,...,numOptions-1
    def computeHumanLoss(self,params,probabilities,chosen_action,lossType,modelName):
        current_models = ["nb","nb-pt","cent","cent-pt"]
        if (modelName not in current_models):
            assert(False)
        prospects= []
        for probability in probabilities:
            prospectSuccess = self.c, probability
            prospectFailure = self.z, 1-probability
            prospects.append((prospectSuccess,prospectFailure))
        pass
        # cpt_vals = evaluateProspectVals(params,prospects)
        # arg = self.softmax(cpt_vals,chosen_action)

        # return -1.*math.log(arg)

    def computeCPTLoss(self,params,probabilities,lossType,modelName):        

        total_loss = 0
        per_question_agent_loss = [None for _ in range(self.numQuestions)]
        per_question_option_loss = [None for _ in range(self.numQuestions)]

        length = len(probabilities)
        start = 0
        if length==self.testSetSize:
            start = self.trainingSetSize

        for question_num in range(length):
            agent_loss  = False
            for is_used in self.machineUsed[start+question_num]:
                if (is_used == True):
                    #compute agent loss
                    agent_loss = True
                    break
            #Here - How to handle consensus not reached case
            if self.actionTaken[start+question_num]==-1:
                continue
            if (agent_loss):
                assert(self.actionTaken[start+question_num] in range(self.teamSize,self.teamSize+self.numAgents))
                per_question_agent_loss[start+question_num] = self.computeAgentLoss(params,probabilities[question_num][self.teamSize:],(self.actionTaken[start+question_num]-self.teamSize),lossType,modelName)
            else:
                assert(self.actionTaken[start+question_num] < len(self.options[start+question_num]))
                per_question_option_loss[start+question_num] = self.computeHumanLoss(params,probabilities[question_num][0:self.teamSize],self.actionTaken[start+question_num],lossType,modelName)
         
        return per_question_option_loss,per_question_agent_loss


    def computeAverageLossPerTeam(self,params, probabilites, lossType, modelName):
        (per_question_option_loss, per_question_agent_loss) = self.computeCPTLoss(params,probabilites,lossType,modelName)
        agent_loss = 0
        option_loss = 0
        agent_count = 0
        option_count = 0
        for (optionLoss,agentLoss) in zip(per_question_option_loss,per_question_agent_loss):
            if (optionLoss != None):
                option_loss += optionLoss
                option_count += 1
            if (agentLoss != None):
                agent_loss += agentLoss
                agent_count += 1

            #If consensus is not reached, it is ignored
            #assert((agentLoss == None) ^ (optionLoss == None))
            assert((agentLoss==None)|(optionLoss== None))
   
        if option_count!=0:
            option_loss /= option_count
        if agent_count!=0:
            agent_loss /= agent_count
            
        return agent_loss + option_loss, option_loss, agent_loss

    
    def chooseCPTParameters(self, probabilities,lossType,modelName):  
        hAlpha, hGamma,hLambda =  (None,None,None)
        hLoss = np.float("Inf") 

        for alpha in np.arange(0.02,1.02,0.02):
            for gamma in np.arange(0.02,1.02,0.02): 
                for lamb in np.arange(0.1,5,0.1):
                    loss,option_loss, agent_loss = self.computeAverageLossPerTeam((alpha,gamma,lamb),probabilities,lossType,modelName) 

                    if (loss<hLoss):
                        hLoss = loss
                        hAlpha = alpha
                        hGamma = gamma
                        hLambda = lamb

        assert(hAlpha != None)
        assert(hGamma != None)
        assert(hLambda != None)
                       
        return (hAlpha, hGamma,hLambda)
    
    def randomModel(self):
        prAgent = len(self.machineAskedQuestions)/self.numQuestions
        prHuman = 1.0-prAgent 
        qi = list()
        for i in range(self.trainingSetSize,self.numQuestions):
            temp = [0.25*prHuman for j in range(0,self.teamSize)]
            for j in range(0,self.teamSize):
                temp.append(0.25*prAgent)
            qi.append(temp)    
        return qi 

    # Agent i has agent i + teamSize
    def getAgentForHuman(self, k):
        return self.teamSize + k
		
if __name__ == '__main__':
    directory = '/home/omid/Datasets/Jeopardy/'
#     cleanEventLog(directory+"event_log.csv")
#     insertInfluenceMatrixNumber(directory+"event_log-Copy.csv")
#     addMissingLogs(directory, directory+"event_log.csv")
    team = pd.read_csv(directory+"team.csv",sep=',',quotechar="|",names=["id","sessionId","roundId", "taskId"])
    nbLoss = list()
    centLoss = list()
    nbPTLoss = list()
    centPTLoss = list()
    randomLoss = list()
    nbAlpha = list()
    nbGamma = list()
    centAlpha = list()
    centGamma = list()
    #lossType = "logit"
    lossType = "softmax"
    nbOptionLoss = list()
    nbAgentLoss = list()
    centOptionLoss = list()
    centAgentLoss = list()
    nbPTOptionLoss = list()
    nbPTAgentLoss = list()
    centPTOptionLoss = list()
    centPTAgentLoss = list()

    testSize = 15
    
    #batchNumbers = [10,84]
    batchNumbers = [10,11,12,13,17,20,21,28,30,33,34,36,37,38,39,41,42,43,44,45,48,49,74,75,77,82,84,85,87,88]
    RATIONAL_PARAMS= (1,1,1)
    NUM_CPT_PARAMS = 2
#     10,11,12,13,17,20,21,28,30,33,34,36,37,38,39,41,42,43,44,45,48,49,74,75,77,82,84,85,87,88

    for i in range(len(team)):  
        if team.iloc[i]['id'] in batchNumbers:
            print("Values of team", team.iloc[i]['id'])
            #create model
            model = HumanDecisionModels(team.iloc[i]['id'], directory)

            rewardsNB1, rewardsCENT1,probabilitiesNB1,probabilitiesCENT1 = model.calculateRewards()
       
            # Compute losses for NB and CENT 
            loss, optionLoss, agentLoss = model.computeAverageLossPerTeam(RATIONAL_PARAMS,probabilitiesNB1[model.trainingSetSize:],lossType,"nb")
            nbOptionLoss.append(optionLoss)
            nbAgentLoss.append(agentLoss)
            nbLoss.append(loss)
            
            loss, optionLoss, agentLoss = model.computeAverageLossPerTeam(RATIONAL_PARAMS,probabilitiesCENT1[model.trainingSetSize:],lossType,"cent")
            centOptionLoss.append(optionLoss)
            centAgentLoss.append(agentLoss)
            centLoss.append(loss)
            	
            # Train alpha,gammma losses for NB-PT
            hAlpha,hGamma,hLambda = model.chooseCPTParameters(probabilitiesNB1,lossType,"nb-pt") 
            print("PT-NB",round(hAlpha,2),round(hGamma,2))
            loss, optionLoss, agentLoss = model.computeAverageLossPerTeam((hAlpha,hGamma,hLambda),probabilitiesNB1[model.trainingSetSize:],lossType,"nb-pt")
            nbPTOptionLoss.append(optionLoss)
            nbPTAgentLoss.append(agentLoss)
            nbPTLoss.append(loss)
            
            # Train alpha,gammma losses for CENT-PT
            hAlpha,hGamma,hLambda = model.chooseCPTParameters(probabilitiesCENT1,lossType,"cent-pt")
            print("CENT-PT",round(hAlpha,2),round(hGamma,2))
            loss, optionLoss, agentLoss = model.computeAverageLossPerTeam((hAlpha,hGamma,hLambda),probabilitiesCENT1[model.trainingSetSize:],lossType,"cent-pt")
            centPTOptionLoss.append(optionLoss)
            centPTAgentLoss.append(agentLoss)
            centPTLoss.append(loss)
        
            # [0.25,0.25,0.25,0.25, 
            #random_prob = [[0.25 for _ in range(model.numAgents+model.teamSize)] for _ in range(model.numQuestions)]
            #randomLoss.append(model.computeCPTLoss(RATIONAL_PARAMS,random_prob,lossType,"random"))

    print("NB1 ",mean(nbLoss),np.std(nbLoss))
    print("CENT1 ",mean(centLoss),np.std(centLoss))
    print("PT-NB-1 ", mean(nbPTLoss),np.std(nbPTLoss))
    print("PT-CENT-1 ",mean(centPTLoss),np.std(centPTLoss))
 
    plt.plot(range(0,len(nbOptionLoss)), nbOptionLoss, label="NB Option Loss")
    plt.plot(range(0,len(nbAgentLoss)), nbAgentLoss, label="NB Agent Loss") 
    plt.plot(range(0,len(nbLoss)), nbLoss, label="NB Total Loss")    
    plt.title("Naive Bayes Model Loss values for all teams")
    plt.legend()
    plt.xlabel("Team")
    plt.ylabel("Loss Value")
    plt.savefig("NB.jpg")
    plt.clf()

    plt.plot(range(0,len(centOptionLoss)), centOptionLoss, label="CENT Option Loss")
    plt.plot(range(0,len(centAgentLoss)), centAgentLoss, label="CENT Agent Loss")   
    plt.plot(range(0,len(centLoss)), centLoss, label="CENT Total Loss")             
    plt.title("CENT Model Loss values for all teams")
    plt.legend()
    plt.xlabel("Team") 
    plt.ylabel("Loss Value")
    plt.savefig("CENT.jpg")
    plt.clf()   

    plt.plot(range(0,len(nbPTOptionLoss)), nbPTOptionLoss, label="NB-PT Option Loss")
    plt.plot(range(0,len(nbPTAgentLoss)), nbPTAgentLoss, label="NB-PT Agent Loss")   
    plt.plot(range(0,len(nbPTLoss)), nbPTLoss, label="NB-PT Total Loss")             
    plt.title("Naive Bayes-PT Model Loss values for all teams")
    plt.legend()
    plt.xlabel("Team") 
    plt.ylabel("Loss Value")
    plt.savefig("NB-PT.jpg")
    plt.clf() 

    plt.plot(range(0,len(centPTOptionLoss)), centPTOptionLoss, label="CENT-PT Option Loss")
    plt.plot(range(0,len(centPTAgentLoss)), centPTAgentLoss, label="CENT-PT Agent Loss")
    plt.plot(range(0,len(centPTLoss)), centPTLoss, label="CENT-PT Total Loss")
    plt.title("CENT-PT Model Loss values for all teams")
    plt.legend()
    plt.xlabel("Team")
    plt.ylabel("Loss Value")
    plt.savefig("CENT-PT.jpg")
    plt.clf()
