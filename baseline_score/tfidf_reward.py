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


def get_tfidf_scores(year,ref_summ):
    corpus_reader = CorpusReader(BASE_DIR)
    peer_summaries = PeerSummaryReader(BASE_DIR)(year)
    mystopwords = set(stopwords.words(LANGUAGE))
    stemmer = PorterStemmer()

    tfidf_vectorizer = TfidfVectorizer(min_df=0)
    tfidf_scores = {}

    for topic,docs,models in tqdm(corpus_reader(year)):
        if '.B' in topic: continue
        all_swos = []
        if ref_summ:
            articles = [' '.join(mm[1]) for mm in models]
        else:
            articles = [' '.join(dd[1]) for dd in docs] 
        swos_articles = [' '.join(sent2stokens_wostop(aa,stemmer,mystopwords,LANGUAGE)) for aa in articles]
        for i,sa in enumerate(swos_articles):
            all_swos.append(sa)

        summaries = peer_summaries[topic]
        sname_list = []
        for ss in summaries:
            sname = ss[0].split('/')[-1]
            if len(ss[1]) == 0: continue
            sname_list.append(sname)
            swos_summ = ' '.join(sent2stokens_wostop(' '.join(ss[1]),stemmer,mystopwords,LANGUAGE))
            all_swos.append(swos_summ)

        vec_matrix = tfidf_vectorizer.fit_transform(all_swos)
        scores = cosine_similarity(vec_matrix[:len(articles)], vec_matrix[len(articles):])
        mean_scores = np.mean(scores,axis=0)
        assert len(mean_scores) == len(sname_list)
        tfidf_scores[topic] = {}
        for i,sname in enumerate(sname_list):
            tfidf_scores[topic][sname] = mean_scores[i]

    return tfidf_scores


def get_human_score(topic, summ_name, human):
    block = summ_name.split('-')[1].split('.')[0]
    id = summ_name.split('.')[-1]
    key = 'topic{}-{}_sum{}'.format(topic.split('.')[0],block,id)
    if key not in human: return None
    else: return human[key]


if __name__ == '__main__':
    year = '08'
    ref_summ = False
    human_metric = 'pyramid'

    print('\n=====year: {}====='.format(year))
    print('=====human score: {}====='.format(human_metric))
    print('=====ref summ: {}====='.format(ref_summ))

    tacData = TacData(BASE_DIR,year)
    human = tacData.getHumanScores('summary', human_metric) # responsiveness or pyramid
    tfidf_scores = get_tfidf_scores(year,ref_summ)
    all_results = {}
    for topic in tfidf_scores:
        if '.B' in topic: continue
        print('\n=====topic {}====='.format(topic))
        human_scores = []
        learnt_scores = []
        for summ_name in tfidf_scores[topic]:
            hscore = get_human_score(topic,summ_name,human)
            if hscore is not None:
                learnt_scores.append(tfidf_scores[topic][summ_name])
                human_scores.append(hscore)
        assert len(human_scores) == len(learnt_scores)
        if len(human_scores) < 2: continue
        results = evaluateReward(learnt_scores,human_scores)
        addResult(all_results,results)
        for kk in results:
            print('{}:\t{}'.format(kk,results[kk]))

    print('\n=====ALL RESULTS=====')
    print('\n=====year: {}====='.format(year))
    print('=====human score: {}====='.format(human_metric))
    print('=====ref summ: {}====='.format(ref_summ))
    for kk in all_results:
        print('{}:\tmax {:.4f}, min {:.4f}, mean {:.4f}, median {:.4f}'.format(kk, np.max(all_results[kk]), np.min(all_results[kk]), np.mean(all_results[kk]), np.median(all_results[kk])))







