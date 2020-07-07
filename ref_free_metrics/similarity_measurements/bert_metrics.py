import sys
sys.path.append('../..')

from sentence_transformers import SentenceTransformer
from transformers import *
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.corpus import stopwords
from tqdm import tqdm
import torch

from resources import BASE_DIR, LANGUAGE
from summariser.data_processor.corpus_reader import CorpusReader
from summariser.data_processor.sys_summ_reader import PeerSummaryReader
from ref_free_metrics.similarity_scorer import parse_documents
from utils import get_human_score
from summariser.data_processor.human_score_reader import TacData
from summariser.utils.evaluator import evaluateReward, addResult
from summariser.utils.data_helpers import sent2stokens_wostop, sent2tokens_wostop


def get_token_vecs(model, tokenizer, sent_list, gpu, sim_metric):
    token_vecs = None

    for i in range(0,len(sent_list),5):
        ss = ' '.join(sent_list[i:i+5])
        if 'roberta' in sim_metric: ss = '<s>' + ss + '</s>'
        tokens = tokenizer.encode(ss)
        tokens = torch.tensor(tokens).unsqueeze(0)
        if gpu:
            tokens = tokens.to('cuda')
        vv = model(tokens)[0][0].data.cpu().numpy()
        if token_vecs is None: token_vecs = vv
        else: token_vecs = np.vstack((token_vecs,vv))
    return  token_vecs


def get_bert_vec_similarity(model, tokenizer, all_sents, ref_num, gpu, sim_metric):
    vec_matrix = []
    non_idx = []
    for i,doc in enumerate(all_sents):
        if len(doc) == 0:
            non_idx.append(i)
            #if 'albert' in sim_metric: vec_matrix.append([0.]*4096)
            #else: vec_matrix.append([0.]*1024)
            vec_matrix.append([0.]*1024)
            continue
        token_vecs = get_token_vecs(model, tokenizer, doc, gpu, sim_metric)
        vec_matrix.append(np.mean(token_vecs,axis=0))
    sim_matrix = cosine_similarity(vec_matrix[ref_num:], vec_matrix[:ref_num])
    scores = np.mean(sim_matrix,axis=1)
    return [ss if j+ref_num not in non_idx else None for j,ss in enumerate(scores)]


def run_bert_vec_metrics(year, ref_metric, sim_metric, gpu=True):
    print('year: {}, ref_metric: {}, sim_metric: {}'.format(year,ref_metric,sim_metric))

    corpus_reader = CorpusReader(BASE_DIR)
    peer_summaries = PeerSummaryReader(BASE_DIR)(year)
    tacData = TacData(BASE_DIR,year)
    human = tacData.getHumanScores('summary', 'pyramid') # responsiveness or pyramid
    sbert_model = SentenceTransformer('bert-large-nli-mean-tokens')#'bert-large-nli-stsb-mean-tokens')

    if sim_metric == 'bert':
        berttokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        bertmodel = BertModel.from_pretrained('bert-large-uncased')
    elif sim_metric == 'albert':
        berttokenizer = AlbertTokenizer.from_pretrained('albert-large-v2')
        bertmodel = AlbertModel.from_pretrained('albert-large-v2')
    else:
        assert 'roberta' in sim_metric
        if 'nli' in sim_metric:
            berttokenizer = RobertaTokenizer.from_pretrained('roberta-large-mnli')
            bertmodel = RobertaModel.from_pretrained('roberta-large-mnli')
        if 'openai' in sim_metric:
            berttokenizer = RobertaTokenizer.from_pretrained('roberta-large-openai-detector')
            bertmodel = RobertaModel.from_pretrained('roberta-large-openai-detector')
        else:
            berttokenizer = RobertaTokenizer.from_pretrained('roberta-large')
            bertmodel = RobertaModel.from_pretrained('roberta-large')
    if gpu: bertmodel.to('cuda')

    mystopwords = set(stopwords.words(LANGUAGE))
    stemmer = PorterStemmer()

    all_results = {}

    # use mover-score to compute scores
    for topic,docs,models in corpus_reader(year):
        if '.B' in topic: continue
        print('\n=====Topic {}====='.format(topic))
        all_sents = []
        if ref_metric == 'true_ref':
            sent_info_dic, sent_vecs, sents_weights = parse_documents(models,sbert_model,ref_metric)
        else:
            sent_info_dic, sent_vecs, sents_weights = parse_documents(docs,sbert_model,ref_metric)
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
        pss = get_bert_vec_similarity(bertmodel,berttokenizer,all_sents,len(ref_sources),gpu,sim_metric)
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
    print('year: {}, ref_metric: {}, sim_metric: bert'.format(year,ref_metric))
    for kk in all_results:
        if kk.startswith('p_'): continue
        print('{}:\tmax {:.4f}, min {:.4f}, mean {:.4f}, median {:.4f}, significant {} out of {}'.format(kk, np.max(all_results[kk]), np.min(all_results[kk]), np.mean(all_results[kk]), np.median(all_results[kk]), len([p for p in all_results['p_{}'.format(kk)] if p<0.05]), len(all_results[kk])))



if __name__ == '__main__':
    run_bert_vec_metrics(year='08', ref_metric='top12_1', sim_metric='bert')

