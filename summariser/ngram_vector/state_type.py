from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import utils.data_helpers as util
import numpy as np
import operator as op
import functools

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from resources import *

#from resources import BASE_DIR,ROUGE_DIR

class State:

    def __init__(self, sum_token_length, base_length, sent_num, block_num, language):
        # hyper parameters
        self.reward_lambda = 0.9

        # learning arguments
        if sum_token_length != None:
            self.sum_token_length = sum_token_length
        else:
            self.sum_token_length = 99999

        self.state_length_computer = StateLengthComputer(block_num,base_length,sent_num)
        self.vec_length = self.state_length_computer.getTotalLength()
        self.summary_vector_length = self.state_length_computer.getStatesLength(block_num)
        self.language = language

        # stemmers and stop words list
        self.stemmer = SnowballStemmer(self.language)
        self.stoplist = set(stopwords.words(self.language))

        # class variables
        #self.draft_summary = ''
        self.draft_summary_list = []
        self.historical_actions = []
        self.available_sents = [i for i in range(0,sent_num+1)]
        self.terminal_state = 0 # 0 stands for non-terminal, and 1 stands for terminal
        self.draft_summary_length = 0

        #some flags/options
        self.newReward = False

    def getSelfVector(self, top_ngrams, sentences):
        return self.getStateVector(self.draft_summary_list,self.historical_actions,
                                   top_ngrams, sentences)

    def getStateVector(self, draft_list, draft_index_list, top_ngrams, sentences):
        '''
        Represent the current draft summary using a vector
        :param draft_list: a list of sentences, included in the current draft summary
        :param draft_index_list: the indices of the included sentences
        :param top_ngrams: top n-grams for all the original documents
        :param sentences: all sentences information, used to find positions information
        :param tfidf: decides to use the Japan version (state_type==True) or the REAPER version
        :return: an numpy array, the vector representation of the state
        '''

        # for empty or over-length draft, return a full-zero vector
        draft_length = 0
        for sent in draft_list:
            draft_length += len(sent.split(' '))

        if len(draft_list) == 0 or draft_list == None: #or draft_length>self.sum_token_length:
            return np.zeros(self.vec_length)

        vector = [0] * self.vec_length
        coverage_num = 0
        redundant_count = 0

        sent_num = len(draft_index_list)
        index = -1 + self.state_length_computer.getIndexUntilSentNum(sent_num)

        draft_ngrams = util.extract_ngrams_count(draft_list,self.stemmer,self.language,self.stoplist)

        num = self.state_length_computer.getStatesLength(sent_num)-5
        for i in range(num):
            index += 1
            if top_ngrams[i] in draft_ngrams:
                vector[index] = 1
                coverage_num += 1
                if draft_ngrams[top_ngrams[i]] >= 2:
                    redundant_count += 1 # draft_ngrams[top_ngrams[i]]-1

        #this is needed, because the above loop does not perform the last add
        index += 1

        #second part: coverage ratio
        vector[index] = coverage_num*1.0/len(top_ngrams)
        index += 1

        #third part: redundant ratio;
        vector[index] = redundant_count*1.0/len(top_ngrams)
        index += 1

        #fourth part: length ratio
        vector[index] = draft_length*1.0/self.sum_token_length
        index += 1

        vector[index] = self.getPositionInfo(sentences,draft_index_list)
        index += 1

        #sixth part: length violation bit
        if draft_length <= self.sum_token_length:
            vector[index] = 1

        if sent_num >= self.state_length_computer.block_num:
            assert index == -1 + self.state_length_computer.getTotalLength()
        else:
            assert index == -1 + self.state_length_computer.getIndexUntilSentNum(sent_num+1)

        return np.array(vector)

    def getPositionInfo(self, sentences, draft_index_list):
        position_index = 0
        for idx in draft_index_list:
            pos = sentences[idx].position
            position_index += 1.0/pos
        return position_index


    def noCommonTokens(self, token_list1, token_list2, word_num_limit=float("inf")):
        # we do not check long sentences
        if len(token_list1) <= word_num_limit and len(token_list2) <= word_num_limit:
            if set(token_list1).isdisjoint(token_list2):
                return True
            else:
                return False
        else:
            return False


    #the Japan version
    def getSimilarity(self, tokens1, sentences2, fullDoc=False):
        # tokens1 is a string of tokens
        # sentences2 is a list of sentences, and each sentences is a list of tokens
        token_list1 = tokens1.split(' ')
        token_str1 = tokens1
        if fullDoc:
            token_str2  = ' '.join(sentences2)
            token_list2 = token_str2.split(' ')
        else:
            token_list2 = sentences2.split(' ')
            token_str2 = sentences2
        tfidf_vectorizer = TfidfVectorizer(min_df=0)
        #print('token_list 1: '+' '.join(token_list1))
        #print('token_list 2: '+' '.join(token_list2))
        if self.noCommonTokens(token_list1,token_list2):
            return 0
        if token_str2 == token_str1:
            return 1
        try:
            tfidf_matrix_train = tfidf_vectorizer.fit_transform([token_str1, token_str2])
        except ValueError:
            return 0
        return cosine_similarity(tfidf_matrix_train[0], tfidf_matrix_train[1])[0][0]
        

    def getNewStateVec(self, new_sent_id, top_ngrams, sentences):
        temp_draft_summary_list = self.draft_summary_list+[sentences[new_sent_id].untokenized_form]
        draft_index_list = self.historical_actions+[new_sent_id]
        return self.getStateVector(temp_draft_summary_list,draft_index_list,top_ngrams,sentences)

    def removeOverlengthSents(self, sents):
        new_avai_acts = [0]
        for sent_id in self.available_sents:
            if sent_id == 0:
                continue
            if len(sents[sent_id-1].untokenized_form.split(' ')) > self.sum_token_length-self.draft_summary_length:
                #self.available_sents.remove(sent_id)
                continue
            else:
                new_avai_acts.append(sent_id)
        self.available_sents = new_avai_acts[:]
        del new_avai_acts


    def updateState(self, new_sent_id, sents, read=False):
        self.draft_summary_list.append(sents[new_sent_id].untokenized_form)
        self.historical_actions.append(new_sent_id)
        self.draft_summary_length += len(sents[new_sent_id].untokenized_form.split(' '))
        if not read:
            self.available_sents.remove(new_sent_id+1)
            self.removeOverlengthSents(sents)
            if self.draft_summary_length > self.sum_token_length:
                self.available_sents = [0]
                self.terminal_state = 1
                print('overlength! should not happen')
                return -1
        return 0

class StateLengthComputer():
    def __init__(self, block_num, base_length, sent_num):
        self.block_num = block_num
        self.lengths = []
        base_num = np.log10(self.ncr(sent_num,1))
        for i in range(block_num):
            self.lengths.append(int(base_length*np.log10(self.ncr(sent_num,i+1))*1.0/base_num)+5)

    def getStatesLength(self, sent_num):
        if sent_num < self.block_num:
            return self.lengths[sent_num-1]
        else:
            return self.lengths[-1]

    def getIndexUntilSentNum(self,n):
        idx = 0
        nn = min(n,self.block_num)
        for i in range(0,nn-1):
            idx += self.getStatesLength(i+1)
        return idx

    def getTotalLength(self):
        return sum(self.lengths)

    def ncr(self, n, r):
        r = min(r, n - r)
        if r == 0: return 1
        numer = functools.reduce(op.mul, range(n, n - r, -1))
        denom = functools.reduce(op.mul, range(1, r + 1))
        return numer // denom


if __name__ == '__main__':
    block_num = 5
    base_num = 80
    sent_num = 400
    print('block num: {}; sentence num: {}; '
          'the summary of length 1 will have {}-dimension states.'.format(block_num, sent_num, base_num))
    com = StateLengthComputer(block_num, base_num, sent_num)
    print('each state length:')
    for i in range(1,9):
        print(com.getStatesLength(i))
    print('starting index:')
    for i in range(1,9):
        print(com.getIndexUntilSentNum(i))
    print('total length:{}'.format(com.getTotalLength()))

















