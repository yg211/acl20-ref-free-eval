from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import random
from tqdm import tqdm

from summariser.ngram_vector.base import Sentence
from summariser.ngram_vector.state_type import *
from utils.data_helpers import *

# from summariser.utils.data_helpers import *

class Vectoriser:

    def __init__(self,docs,sum_len=100,no_stop_words=True,stem=True,block=1,base=200,lang='english'):
        self.docs = docs
        self.without_stopwords = no_stop_words
        self.stem = stem
        self.block_num = block
        self.base_length = base
        self.language = lang
        self.sum_token_length = sum_len

        self.stemmer = PorterStemmer()
        self.stoplist = set(stopwords.words(self.language))
        self.sim_scores = {}
        self.stemmed_sentences_list = []

        self.load_data()

    def sample_random_summaries(self,num):
        act_list = []

        for ii in tqdm(range(num), desc='generating samples for memory replay'):
            state = State(self.sum_token_length, self.base_length, len(self.sentences),self.block_num, self.language)
            while state.available_sents != [0]:
                new_id = random.choice(state.available_sents)
                if new_id == 0: continue
                if new_id > 0 and len(self.sentences[new_id-1].untokenized_form.split(' ')) > self.sum_token_length: continue
                state.updateState(new_id-1,self.sentences)
            actions = state.historical_actions
            act_list.append(actions)
        return act_list


    def getSummaryVectors(self,summary_acts_list):
        vector_list = []

        for act_list in summary_acts_list:
            state = State(self.sum_token_length, self.base_length, len(self.sentences), self.block_num, self.language)
            for i, act in enumerate(act_list):
                state.updateState(act, self.sentences, read=True)
            vector = state.getSelfVector(self.top_ngrams_list, self.sentences)
            vector_list.append(vector)

        return vector_list

    def sent2tokens(self, sent_str):
        if self.without_stopwords and self.stem:
            return sent2stokens_wostop(sent_str, self.stemmer, self.stoplist, self.language)
        elif self.without_stopwords == False and self.stem:
            return sent2stokens(sent_str, self.stemmer, self.language)
        elif self.without_stopwords and self.stem == False:
            return sent2tokens_wostop(sent_str, self.stoplist, self.language)
        else:  # both false
            return sent2tokens(sent_str, self.language)


    def load_data(self):
        self.sentences = []
        for doc_id, doc in enumerate(self.docs):
            doc_name, doc_sents = doc
            doc_tokens_list = []
            for sent_id, sent_text in enumerate(doc_sents):
                token_sent = word_tokenize(sent_text, self.language)
                current_sent = Sentence(token_sent, doc_id, sent_id + 1)
                untokenized_form = untokenize(token_sent)
                current_sent.untokenized_form = untokenized_form
                current_sent.length = len(untokenized_form.split(' '))
                self.sentences.append(current_sent)
                sent_tokens = self.sent2tokens(untokenized_form)
                doc_tokens_list.extend(sent_tokens)
                stemmed_form = ' '.join(sent_tokens)
                self.stemmed_sentences_list.append(stemmed_form)
        #print('total sentence num: ' + str(len(self.sentences)))

        self.state_length_computer = StateLengthComputer(self.block_num, self.base_length, len(self.sentences))
        self.top_ngrams_num = self.state_length_computer.getStatesLength(self.block_num)
        self.vec_length = self.state_length_computer.getTotalLength()

        sent_list = []
        for sent in self.sentences:
            sent_list.append(sent.untokenized_form)
        self.top_ngrams_list = getTopNgrams(sent_list, self.stemmer, self.language,
                                            self.stoplist, 2, self.top_ngrams_num)


