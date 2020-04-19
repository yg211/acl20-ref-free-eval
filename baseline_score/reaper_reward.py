import sys
sys.path.append('../')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np

from resources import LANGUAGE,BASE_DIR
from summariser.data_processor.corpus_reader import CorpusReader
from summariser.data_processor.sys_summ_reader import PeerSummaryReader
from summariser.data_processor.human_score_reader import TacData
from summariser.utils.evaluator import evaluateReward, addResult
from summariser.utils.data_helpers import sent2stokens_wostop


class ReaperMetricGenerator:
    def __init__(self,docs):
        self.stopwords = set(stopwords.words(LANGUAGE))
        self.stemmer = PorterStemmer()

        self.docs = docs
        self.sentences = []
        for doc in docs:
            self.sentences.extend(doc)
        self.stem_wostop_sents = [' '.join(sent2stokens_wostop(ss,self.stemmer,self.stopwords, 'english')) for ss in self.sentences]
        self.tfidf_vectorizer = TfidfVectorizer(min_df=0)

    def __call__(self, summ_sents):
        stemed_wostop_summ_sents = [sent2stokens_wostop(ss,self.stemmer, self.stopwords, 'english') for ss in summ_sents]
        sim_score = 0
        red_socre = 0
        for i,ss in enumerate(stemed_wostop_summ_sents):
            sim_score += self._get_similarity(ss, self.stem_wostop_sents)
            for j in range(i):
                red_socre += self._get_similarity(ss,self.stem_wostop_sents)

        return sim_score*0.9 - red_socre*0.1

    def _get_similarity(self, text1, text2):
        tfidf_matrix_train = self.tfidf_vectorizer.fit_transform([' '.join(text1), ' '.join(text2)])
        return cosine_similarity(tfidf_matrix_train[0], tfidf_matrix_train[1])[0][0]


def get_reaper_scores(year):
    corpus_reader = CorpusReader(BASE_DIR)
    peer_summaries = PeerSummaryReader(BASE_DIR)(year)
    reaper_scores = {}

    for topic,docs,models in tqdm(corpus_reader(year)):
        if '.B' in topic: continue
        reaper_scores[topic] = {}
        articles = [dd[1] for dd in docs] 
        reaper_agent = ReaperMetricGenerator(articles)

        summaries = peer_summaries[topic]
        for ss in summaries:
            sname = ss[0].split('/')[-1]
            if len(ss[1]) == 0: continue
            rpv = reaper_agent(ss[1])
            reaper_scores[topic][sname] = rpv

    return reaper_scores


def get_human_score(topic, summ_name, human):
    block = summ_name.split('-')[1].split('.')[0]
    id = summ_name.split('.')[-1]
    key = 'topic{}-{}_sum{}'.format(topic.split('.')[0],block,id)
    if key not in human: return None
    else: return human[key]


if __name__ == '__main__':
    year = '08'
    human_metric = 'pyramid'

    print('\n=====year: {}====='.format(year))
    print('=====human score: {}====='.format(human_metric))

    tacData = TacData(BASE_DIR,year)
    human = tacData.getHumanScores('summary', human_metric) # responsiveness or pyramid
    reaper_scores = get_reaper_scores(year)
    all_results = {}
    for topic in reaper_scores:
        if '.B' in topic: continue
        print('\n=====topic {}====='.format(topic))
        human_scores = []
        learnt_scores = []
        for summ_name in reaper_scores[topic]:
            hscore = get_human_score(topic,summ_name,human)
            if hscore is not None:
                learnt_scores.append(reaper_scores[topic][summ_name])
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







