import sys
import os

sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('./'))

from summariser.ngram_vector.state_type import State
#from summariser.utils.summary_samples_reader import *
from utils.misc import softmax_sample, normaliseList
from summariser.ngram_vector.vector_generator import Vectoriser

import numpy as np
import random
from tqdm import tqdm

import torch
from torch.autograd import Variable


class DeepTDAgent:
    def __init__(self, vectoriser, summaries, train_round=5000, strict_para=3, gpu=True):

        # hyper parameters
        self.gamma = 1.
        self.epsilon = 1.
        self.alpha = 0.001
        self.lamb = 1.0

        # training options and hyper-parameters
        self.train_round = train_round
        self.strict_para = strict_para

        # samples
        self.summaries = summaries
        self.vectoriser = vectoriser
        self.weights = np.zeros(self.vectoriser.vec_length)

        # deep training
        self.hidden_layer_width = int(self.vectoriser.vec_length/2)
        self.gpu = gpu


    def __call__(self,reward_list,normalise=True):
        self.softmax_list = []
        if normalise: rewards = normaliseList(reward_list)
        else: rewards = reward_list
        summary = self.trainModel(self.summaries, rewards)
        return summary


    def trainModel(self, summary_list, reward_list):
        _,self.softmax_list = softmax_sample(reward_list,self.strict_para,[],True)

        self.deep_model = torch.nn.Sequential(
            torch.nn.Linear(self.vectoriser.vec_length, self.hidden_layer_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_layer_width, self.hidden_layer_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_layer_width, 1),
        )
        if self.gpu: self.deep_model.to('cuda')
        self.optimiser = torch.optim.Adam(self.deep_model.parameters())

        for ii in tqdm(range(int(self.train_round)), desc='neural-rl training episodes'):

            # if (ii+1)%1000 == 0 and ii!= 0:
            #     print('trained for {} episodes.'.format(ii+1))

            #find a new sample, using the softmax value
            idx = softmax_sample(reward_list,self.strict_para,self.softmax_list,False)

            # train the model for one episode
            loss = self.train(summary_list[idx],reward_list[idx])

        summary = self.produceSummary()
        return ' '.join(summary)

    def produceSummary(self):
        state = State(self.vectoriser.sum_token_length, self.vectoriser.base_length, len(self.vectoriser.sentences),self.vectoriser.block_num, self.vectoriser.language)

        # select sentences greedily
        while state.terminal_state == 0:
            new_sent_id = self.getGreedySent(state)
            if new_sent_id == 0:
                break
            else:
                state.updateState(new_sent_id-1, self.vectoriser.sentences) #, production=True)

        return state.draft_summary_list[:]

    def coreUpdate(self, reward, current_vec, new_vec, vector_e):
        delta = reward + np.dot(self.weights, self.gamma * new_vec - current_vec)
        vec_e = self.gamma * self.lamb * vector_e + current_vec
        self.weights = self.weights + self.alpha * delta * vec_e
        return vec_e

    def getGreedySent(self, state):
        vec_list = []
        str_vec_list = []

        for act_id in state.available_sents:

            # for action 'finish', the reward is the terminal reward
            if act_id == 0:
                current_state_vec = state.getSelfVector(self.vectoriser.top_ngrams_list, self.vectoriser.sentences)
                vec_variable = Variable(torch.from_numpy(np.array(current_state_vec)).float())
                if self.gpu: vec_variable = vec_variable.to('cuda')
                terminate_reward = self.deep_model(vec_variable.unsqueeze(0)).data.cpu().numpy()[0][0]

            # otherwise, the reward is 0, and value-function can be computed using the weight
            else:
                temp_state_vec = state.getNewStateVec(act_id-1, self.vectoriser.top_ngrams_list,
                                                      self.vectoriser.sentences)
                vec_list.append(temp_state_vec)
                str_vec = ''
                for ii,vv in enumerate(temp_state_vec):
                    if vv != 0.:
                        str_vec += '{}:{};'.format(ii,vv)
                str_vec_list.append(str_vec)

        if len(vec_list) == 0: return 0
        # get action that results in highest values
        variable = Variable(torch.from_numpy(np.array(vec_list)).float())
        if self.gpu: variable = variable.to('cuda')
        values = self.deep_model(variable)
        values_list = values.data.cpu().numpy()
        values_list = [v[0] for v in values_list]
        #print('vectors list: ')
        #for vv in str_vec_list:
            #print(vv)
        max_value = float('-inf')
        max_idx = -1
        for ii,value in enumerate(values_list):
            if value > max_value:
                max_value = value
                max_idx = ii

        if terminate_reward > max_value:
            return 0
        else:
            return state.available_sents[max_idx+1]


    def train(self,summary_actions, summary_value):
        state = State(self.vectoriser.sum_token_length, self.vectoriser.base_length,
                      len(self.vectoriser.sentences), self.vectoriser.block_num, self.vectoriser.language)
        current_vec = state.getStateVector(state.draft_summary_list, state.historical_actions,
                                           self.vectoriser.top_ngrams_list, self.vectoriser.sentences)

        vec_list = []
        vec_list.append(current_vec)

        for act in summary_actions:
            #BE CAREFUL: here the argument for updateState is act, because here act is the real sentence id, not action
            reward = state.updateState(act, self.vectoriser.sentences, True)
            new_vec = state.getStateVector(state.draft_summary_list, state.historical_actions,
                                           self.vectoriser.top_ngrams_list,self.vectoriser.sentences)
            vec_list.append(new_vec)

        return self.deepTrain(vec_list,summary_value)


    def deepTrain(self, vec_list, last_reward):
        vecs = Variable(torch.from_numpy(np.array(vec_list)).float())
        if self.gpu: vecs = vecs.to('cuda')
        value_variables = self.deep_model(vecs)
        #print('value var', value_variables)
        value_list = value_variables.data.cpu().numpy()
        target_list = []
        for idx in range(len(value_list)-1):
            target_list.append(self.gamma*value_list[idx+1][0])
        target_list.append(last_reward)
        target_variables = Variable(torch.from_numpy(np.array(target_list)).float()).unsqueeze(0).view(-1,1)
        #print('target var', target_variables)

        loss_fn = torch.nn.MSELoss()
        if self.gpu:
            #value_variables = value_variables.to('cuda')
            target_variables = target_variables.to('cuda')

        loss = loss_fn(value_variables,target_variables)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss.item()



