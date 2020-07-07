import sys
sys.path.append('../..')

from sentence_transformers import SentenceTransformer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.corpus import stopwords
from tqdm import tqdm
from allennlp.modules.elmo import Elmo, batch_to_ids

from resources import BASE_DIR, LANGUAGE
from summariser.data_processor.corpus_reader import CorpusReader
from summariser.data_processor.sys_summ_reader import PeerSummaryReader
from ref_free_metrics.similarity_scorer import parse_documents
from utils import get_human_score
from summariser.data_processor.human_score_reader import TacData
from summariser.utils.evaluator import evaluateReward, addResult
from summariser.utils.data_helpers import sent2stokens_wostop, sent2tokens_wostop


options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"


def get_elmo_vec_similarity(elmo_model, all_sents, ref_num):
    vec_matrix = []
    non_idx = []
    for i,doc in enumerate(all_sents):
        tokenized_sents = [word_tokenize(ss) for ss in doc]
        if len(tokenized_sents) == 0:
            non_idx.append(i)
            vec_matrix.append([0.]*1024)
            continue
        sent_length = [len(ts) for ts in tokenized_sents]
        character_ids = batch_to_ids(tokenized_sents).to('cuda')
        elmo_vecs = elmo_model(character_ids)['elmo_representations']
        token_vecs = None
        for j in range(len(sent_length)):
            vv = 0.5*elmo_vecs[0][j][:sent_length[j]]+0.5*elmo_vecs[1][j][:sent_length[j]]
            vv = vv.data.cpu().numpy()
            if token_vecs is None: token_vecs = vv
            else: token_vecs = np.vstack((token_vecs, vv))
        vec_matrix.append(np.mean(token_vecs,axis=0))
    sim_matrix = cosine_similarity(vec_matrix[ref_num:], vec_matrix[:ref_num])
    scores = np.mean(sim_matrix,axis=1)
    return [ss if j+ref_num not in non_idx else None for j,ss in enumerate(scores)]


def run_elmo_vec_metrics(year, ref_metric):
    print('year: {}, ref_metric: {}, sim_metric: elmo'.format(year,ref_metric))

    corpus_reader = CorpusReader(BASE_DIR)
    peer_summaries = PeerSummaryReader(BASE_DIR)(year)
    tacData = TacData(BASE_DIR,year)
    human = tacData.getHumanScores('summary', 'pyramid') # responsiveness or pyramid
    elmo_model = Elmo(options_file, weight_file, 2, dropout=0)
    elmo_model.to('cuda')
    bert_model = SentenceTransformer('bert-large-nli-mean-tokens')#'bert-large-nli-stsb-mean-tokens')

    mystopwords = set(stopwords.words(LANGUAGE))
    stemmer = PorterStemmer()

    all_results = {}

    # use mover-score to compute scores
    for topic,docs,models in corpus_reader(year):
        if '.B' in topic: continue
        print('\n=====Topic {}====='.format(topic))
        all_sents = []
        if ref_metric == 'true_ref':
            sent_info_dic, sent_vecs, sents_weights = parse_documents(models,bert_model,ref_metric)
        else:
            sent_info_dic, sent_vecs, sents_weights = parse_documents(docs,bert_model,ref_metric)
        ref_dic = {k:sent_info_dic[k] for k in sent_info_dic if sents_weights[k]>=0.1}
        ref_sources = set(ref_dic[k]['doc'] for k in ref_dic)
        if len(ref_dic) >= 15:
            for rs in ref_sources:
                ref_sents = [ref_dic[k]['text'] for k in ref_dic if ref_dic[k]['doc']==rs]
                all_sents.append(ref_sents)
        else:
            ref_sents = [ref_dic[k]['text'] for k in ref_dic]
            all_sents.append(ref_sents)
            ref_sources = [1]
        for ss in peer_summaries[topic]:
            all_sents.append(ss[1])
        # compute word-vec-cosine score
        pss = get_elmo_vec_similarity(elmo_model,all_sents,len(ref_sources))
        # compute correlation
        hss = [get_human_score(topic,ss[0].split('/')[-1],human) for ss in peer_summaries[topic]]
        pseudo_scores, human_scores = [], []
        for i in range(len(pss)):
            if hss[i] is not None and pss[i] is not None:
                pseudo_scores.append(pss[i])
                human_scores.append(hss[i])
        assert len(human_scores) == len(pseudo_scores)
        if len(human_scores) < 2: continue
        results = evaluateReward(pseudo_scores,human_scores)
        addResult(all_results,results)
        for kk in results:
            print('{}:\t{}'.format(kk,results[kk]))

    print('\n=====ALL RESULTS=====')
    print('year: {}, ref_metric: {}, sim_metric: elmo'.format(year,ref_metric))
    for kk in all_results:
        print('{}:\tmax {:.4f}, min {:.4f}, mean {:.4f}, median {:.4f}'.format(kk, np.max(all_results[kk]), np.min(all_results[kk]), np.mean(all_results[kk]), np.median(all_results[kk])))



if __name__ == '__main__':
    run_elmo_vec_metrics(year='08', ref_metric='top12_1')


