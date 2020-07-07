import sys
sys.path.append('../..')

from sentence_transformers import SentenceTransformer
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.corpus import stopwords

from resources import BASE_DIR, LANGUAGE
from summariser.data_processor.corpus_reader import CorpusReader
from summariser.data_processor.sys_summ_reader import PeerSummaryReader
from ref_free_metrics.similarity_scorer import parse_documents
from utils import get_human_score
from summariser.data_processor.human_score_reader import TacData
from summariser.utils.evaluator import evaluateReward, addResult


def get_idf(doc_token_list):
    df_dic = {}
    for i,doc_tokens in enumerate(doc_token_list):
        if doc_tokens is None: continue
        for tk in doc_tokens:
            if tk in df_dic: df_dic[tk].append(i)
            else: df_dic[tk] = [i]

    doc_num = len(doc_token_list)
    idf_list = []
    for i,doc_tokens in enumerate(doc_token_list):
        if doc_tokens is None:
            idf_list.append(None)
            continue
        idf = []
        for tk in doc_tokens: idf.append(-1.*np.log( (len(set(df_dic[tk]))+0.5)/(doc_num+0.5)))
        idf_list.append(idf)
    return np.array(idf_list)


def get_sbert_score(ref_token_vecs, ref_tokens, summ_token_vecs, summ_tokens, sim_metric):
    recall_list = []
    precision_list = []
    f1_list = []
    empty_summs_ids = []

    if 'idf' in sim_metric:
        ref_idf = get_idf(ref_tokens)
        summ_idf = get_idf(summ_tokens)

    for i,rvecs in enumerate(ref_token_vecs):
        r_recall_list = []
        r_precision_list = []
        r_f1_list = []
        for j,svecs in enumerate(summ_token_vecs):
            if svecs is None:
                empty_summs_ids.append(j)
                r_recall_list.append(None)
                r_precision_list.append(None)
                r_f1_list.append(None)
                continue
            sim_matrix = cosine_similarity(rvecs,svecs)
            if 'idf' in sim_metric:
                idf_recall = np.dot(np.max(sim_matrix, axis=1), ref_idf[i])
                idf_precision = np.dot(np.max(sim_matrix, axis=0), summ_idf[j])
                idf_f1 = 2. * idf_recall * idf_precision / (idf_recall + idf_precision)
                r_recall_list.append(idf_recall)
                r_precision_list.append(idf_precision)
                r_f1_list.append(idf_f1)
            else:
                recall = np.mean(np.max(sim_matrix, axis=1))
                precision = np.mean(np.max(sim_matrix, axis=0))
                f1 = 2. * recall * precision / (recall + precision)
                r_recall_list.append(recall)
                r_precision_list.append(precision)
                r_f1_list.append(f1)
        recall_list.append(r_recall_list)
        precision_list.append(r_precision_list)
        f1_list.append(r_f1_list)
    empty_summs_ids = list(set(empty_summs_ids))
    recall_list = np.array(recall_list)
    precision_list = np.array(precision_list)
    f1_list = np.array(f1_list)
    if 'recall' in sim_metric:
        scores = []
        for i in range(len(summ_token_vecs)):
            if i in empty_summs_ids: scores.append(None)
            else: scores.append(np.mean(recall_list[:,i]))
        return scores
        #return np.mean(np.array(recall_list), axis=0)
    elif 'precision' in sim_metric:
        scores = []
        for i in range(len(summ_token_vecs)):
            if i in empty_summs_ids: scores.append(None)
            else: scores.append(np.mean(precision_list[:,i]))
        return scores
        #return np.mean(np.array(precision_list), axis=0)
    else:
        assert 'f1' in sim_metric
        scores = []
        for i in range(len(summ_token_vecs)):
            if i in empty_summs_ids: scores.append(None)
            else: scores.append(np.mean(f1_list[:,i]))
        return scores
        #return np.mean(np.mean(f1_list),axis=0)


def get_token_vecs(model,sents,remove_stopwords=True):
    if len(sents) == 0: return None, None
    vecs, tokens = model.encode(sents, token_vecs=True)
    for i, rtv in enumerate(vecs):
        if i==0:
            full_vec = rtv
            full_token = tokens[i]
        else:
            full_vec = np.row_stack((full_vec, rtv))
            full_token.extend(tokens[i])
    if remove_stopwords:
        mystopwords = list(set(stopwords.words(LANGUAGE)))
        mystopwords.extend(['[cls]','[sep]'])
        wanted_idx = [j for j,tk in enumerate(full_token) if tk.lower() not in mystopwords]
    else:
        wanted_idx = [k for k in range(len(full_token))]
    return full_vec[wanted_idx], np.array(full_token)[wanted_idx]


def run_sbert_score_metrics(year, ref_metric, sim_metric):
    print('year: {}, ref_metric: {}, sim_metric: sbert-{}'.format(year,ref_metric,sim_metric))
    sbert_type = sim_metric
    corpus_reader = CorpusReader(BASE_DIR)
    peer_summaries = PeerSummaryReader(BASE_DIR)(year)
    tacData = TacData(BASE_DIR,year)
    human = tacData.getHumanScores('summary', 'pyramid') # responsiveness or pyramid
    bert_model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')  # 'bert-large-nli-stsb-mean-tokens')
    all_results = {}


    # use mover-score to compute scores
    for topic,docs,models in corpus_reader(year):
        if '.B' in topic: continue
        print('\n=====Topic {}====='.format(topic))
        if ref_metric == 'true_ref':
            sent_info_dic, sent_vecs, sents_weights = parse_documents(models,bert_model,ref_metric)
        else:
            sent_info_dic, sent_vecs, sents_weights = parse_documents(docs,bert_model,ref_metric)
        ref_dic = {k:sent_info_dic[k] for k in sent_info_dic if sents_weights[k]>=0.1}
        print('extracted sent ratio', len(ref_dic)*1./len(sent_info_dic))
        ref_sources = set(ref_dic[k]['doc'] for k in ref_dic)
        # get sents in ref/doc
        ref_sents = []
        if len(ref_dic) >= 15: #all(['ref' in rs for rs in ref_sources]):
            for rs in ref_sources:
                ref_sents.append([ref_dic[k]['text'] for k in ref_dic if ref_dic[k]['doc']==rs])
        else:
            ref_sents.append([ref_dic[k]['text'] for k in ref_dic])
        ref_vecs = []
        ref_tokens = []
        for rsents in ref_sents:
            vv, tt = get_token_vecs(bert_model,rsents)
            ref_vecs.append(vv)
            ref_tokens.append(tt)
        # get sents in system summaries
        summ_vecs = []
        summ_tokens = []
        for ss in peer_summaries[topic]:
            vv, tt = get_token_vecs(bert_model,ss[1])
            summ_vecs.append(vv)
            summ_tokens.append(tt)
        # get sbert-score
        pss = get_sbert_score(ref_vecs, ref_tokens, summ_vecs, summ_tokens, sbert_type)
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
    print('year: {}, ref_metric: {}, sim_metric: sbert-{}'.format(year,ref_metric,sim_metric))
    for kk in all_results:
        if kk.startswith('p_'): continue
        print('{}:\tmax {:.4f}, min {:.4f}, mean {:.4f}, median {:.4f}, significant {} out of {}'.format(kk, np.max(all_results[kk]), np.min(all_results[kk]), np.mean(all_results[kk]), np.median(all_results[kk]), len([p for p in all_results['p_{}'.format(kk)] if p<0.05]), len(all_results[kk])))



if __name__ == '__main__':
    run_sbert_score_metrics(year='08', ref_metric='top12_1', sim_metric='f1')

