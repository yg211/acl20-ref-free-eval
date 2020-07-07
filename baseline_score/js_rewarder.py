import sys
sys.path.append('../')
from nltk.stem.porter import PorterStemmer
from collections import OrderedDict
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from summariser.utils.data_helpers import sent2stokens_wostop,extract_ngrams2
from resources import LANGUAGE,BASE_DIR
from tqdm import tqdm
import numpy as np

from summariser.data_processor.corpus_reader import CorpusReader
from summariser.data_processor.sys_summ_reader import PeerSummaryReader
from summariser.utils.misc import jsd,normaliseList
from summariser.data_processor.human_score_reader import TacData
from summariser.utils.evaluator import evaluateReward, addResult


class JSMetricGenerator:
    def __init__(self,docs,nlist=None,wanted='js'):
        self.stopwords = set(stopwords.words(LANGUAGE))
        self.stemmer = PorterStemmer()

        self.docs = docs
        self.sentences = []
        for doc in docs:
            self.sentences.extend(doc)
        if nlist is None:
            self.nlist = [1]
        else:
            self.nlist = nlist
        self.wanted = wanted
        self.doc_dist = self._get_word_distribution(self.sentences,self.nlist)

    def __call__(self, summ_sents):
        summ_dist = self._get_word_distribution(summ_sents,self.nlist)
        doc_dist_list, summ_dist_list = self._build_doc_summ_dist(self.doc_dist.copy(), summ_dist)
        return jsd(doc_dist_list, summ_dist_list, wanted=self.wanted)

    def _build_doc_summ_dist(self, doc_dic, summ_dic, default_smooth=0.1):
        doc_vocab = set(doc_dic.keys())
        summ_vocab = set(summ_dic.keys())
        all_vocab = doc_vocab.union(summ_vocab)

        doc_list = []
        summ_list = []
        for vv in all_vocab:
            if vv in doc_dic: doc_list.append(doc_dic[vv])
            else: doc_list.append(default_smooth)
            if vv in summ_dic: summ_list.append(summ_dic[vv])
            else: summ_list.append(default_smooth)

        return doc_list, summ_list
        
    def _get_word_distribution(self,sent_list,nlist):
        vocab_list = []
        dic = OrderedDict((el,0) for el in vocab_list)

        for sent in sent_list:
            ngrams = []
            for n in nlist:
                if n == 1:
                    ngrams.extend(sent2stokens_wostop(sent,self.stemmer,self.stopwords,LANGUAGE))
                else:
                    ngrams.extend(extract_ngrams2([sent],self.stemmer,LANGUAGE,n))

            for ww in ngrams:
                if ww in dic:
                    dic[ww] = dic[ww]+1
                else:
                    dic[ww] = 1

        return dic


def get_js_scores(year,ngram_list,wanted):
    corpus_reader = CorpusReader(BASE_DIR)
    peer_summaries = PeerSummaryReader(BASE_DIR)(year)
    js_scores = {}

    for topic,docs,models in tqdm(corpus_reader(year)):
        if '.B' in topic: continue
        js_scores[topic] = {}
        articles = [dd[1] for dd in docs] 
        js_agent = JSMetricGenerator(articles, ngram_list, wanted)

        summaries = peer_summaries[topic]
        for ss in summaries:
            sname = ss[0].split('/')[-1]
            if len(ss[1]) == 0: continue
            summary = ss[1]
            jsv = js_agent(summary)
            js_scores[topic][sname] = jsv

    return js_scores


def get_human_score(topic, summ_name, human):
    block = summ_name.split('-')[1].split('.')[0]
    id = summ_name.split('.')[-1]
    key = 'topic{}-{}_sum{}'.format(topic.split('.')[0],block,id)
    if key not in human: return None
    else: return human[key]


if __name__ == '__main__':
    year = '08'
    human_metric = 'pyramid'
    ngram_list = [1]
    wanted = 'js'

    print('\n=====year: {}====='.format(year))
    print('=====human score: {}====='.format(human_metric))
    print('=====system score: {} {}=====\n'.format(wanted,ngram_list))

    tacData = TacData(BASE_DIR,year)
    human = tacData.getHumanScores('summary', human_metric) # responsiveness or pyramid
    js_scores = get_js_scores(year,ngram_list,wanted)
    all_results = {}
    for topic in js_scores:
        if '.B' in topic: continue
        print('\n=====topic {}====='.format(topic))
        human_scores = []
        learnt_scores = []
        for summ_name in js_scores[topic]:
            hscore = get_human_score(topic,summ_name,human)
            if hscore is not None:
                learnt_scores.append(-js_scores[topic][summ_name])
                human_scores.append(hscore)
        assert len(human_scores) == len(learnt_scores)
        if len(human_scores) < 2: continue
        results = evaluateReward(learnt_scores,human_scores)
        addResult(all_results,results)
        for kk in results:
            print('{}:\t{}'.format(kk,results[kk]))

    print('\n=====ALL RESULTS=====')
    for kk in all_results:
        print('{}:\tmax {:.4f}, min {:.4f}, mean {:.4f}, median {:.4f}'.format(kk, np.max(all_results[kk]), np.min(all_results[kk]), np.mean(all_results[kk]), np.median(all_results[kk])))







