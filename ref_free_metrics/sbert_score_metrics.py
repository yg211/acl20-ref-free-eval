import sys
sys.path.append('../')

import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import copy

from resources import BASE_DIR, LANGUAGE
from ref_free_metrics.similarity_scorer import parse_documents

def get_sbert_score(ref_token_vecs, ref_tokens, summ_token_vecs, summ_tokens, sim_metric):
    recall_list = []
    precision_list = []
    f1_list = []
    empty_summs_ids = []

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


def get_token_vecs(sent_idx, all_token_vecs, all_tokens):
    for i,si in enumerate(sent_idx):
        assert len(all_token_vecs[si]) == len(all_tokens[si])
        if i == 0:
            full_vec = copy.deepcopy(all_token_vecs[si])
            full_token = copy.deepcopy(all_tokens[si])
        else:
            full_vec = np.row_stack((full_vec, all_token_vecs[si]))
            full_token.extend(all_tokens[si])
    mystopwords = list(set(stopwords.words(LANGUAGE)))
    mystopwords.extend(['[cls]','[sep]'])
    wanted_idx = [j for j,tk in enumerate(full_token) if tk.lower() not in mystopwords]
    return full_vec[wanted_idx], np.array(full_token)[wanted_idx]


def get_all_token_vecs(model, sent_info_dict):
    all_sents = [sent_info_dict[i]['text'] for i in sent_info_dict]
    all_vecs, all_tokens = model.encode(all_sents, token_vecs=True)
    assert len(all_vecs) == len(all_tokens)
    for i in range(len(all_vecs)):
        assert len(all_vecs[i]) == len(all_tokens[i])
    return all_vecs, all_tokens



def get_rewards(docs, summaries, ref_metric, sim_metric='f1'):
    bert_model = SentenceTransformer('bert-large-nli-stsb-mean-tokens') 
    # word and sentence tokenization
    sent_info_dic, _, sents_weights = parse_documents(docs,None,ref_metric)
    all_token_vecs, all_tokens = get_all_token_vecs(bert_model, sent_info_dic)
    # build pseudo-ref
    ref_dic = {k:sent_info_dic[k] for k in sent_info_dic if sents_weights[k]>=0.1}
    # get sents in the pseudo ref
    ref_sources = set(ref_dic[k]['doc'] for k in ref_dic)
    ref_idxs = []
    if len(ref_dic) >= 15: #all(['ref' in rs for rs in ref_sources]):
        # group sentences from the same doc into one pseudo ref
        for rs in ref_sources:
            ref_idxs.append([k for k in ref_dic if ref_dic[k]['doc']==rs])
    else:
        ref_idxs.append([k for k in ref_dic])
    # get vecs and tokens of the pseudo reference
    ref_vecs = []
    ref_tokens = []
    for ref in ref_idxs:
        vv, tt = get_token_vecs(ref, all_token_vecs, all_tokens)
        ref_vecs.append(vv)
        ref_tokens.append(tt)
    # get vecs and tokens of the summaries
    summ_vecs = []
    summ_tokens = []
    for summ in summaries:
        vv, tt = get_token_vecs(summ, all_token_vecs, all_tokens)
        summ_vecs.append(vv)
        summ_tokens.append(tt)
    # measure similarity between system summaries and the pseudo-ref
    scores = get_sbert_score(ref_vecs, ref_tokens, summ_vecs, summ_tokens, sim_metric)
    return scores


'''

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



def old_get_reward(docs, peer_summaries, ref_metric, sim_metric='f1', mute=True):
    sbert_type = sim_metric
    bert_model = SentenceTransformer('bert-large-nli-stsb-mean-tokens') 
    # word and sentence tokenization
    sent_info_dic, _, sents_weights = parse_documents(docs,None,ref_metric)
    all_token_vecs, all_tokens = get_all_token_vecs(bert_model, sent_info_dic)
    # build pseudo-ref
    ref_dic = {k:sent_info_dic[k] for k in sent_info_dic if sents_weights[k]>=0.1}
    if not mute: print('extracted sent ratio', len(ref_dic)*1./len(sent_info_dic))
    ref_sources = set(ref_dic[k]['doc'] for k in ref_dic)
    # get sents in the pseudo ref
    ref_sents = []
    if len(ref_dic) >= 15: #all(['ref' in rs for rs in ref_sources]):
        # group sentences from the same doc into one pseudo ref
        for rs in ref_sources:
            ref_sents.append([ref_dic[k]['text'] for k in ref_dic if ref_dic[k]['doc']==rs])
    else:
        ref_sents.append([ref_dic[k]['text'] for k in ref_dic])
    # tokenize sentences from pseudo-ref and get their embeddings
    ref_vecs = []
    ref_tokens = []
    for rsent in ref_sents:
        vv, tt = get_token_vecs(bert_model,rsent)
        ref_vecs.append(vv)
        ref_tokens.append(tt)
    # get sents in system summaries
    summ_vecs = []
    summ_tokens = []
    for ss in peer_summaries:
        summ_vecs.append( [sent_vecs[sid] for sid in ss] )
        summ_tokens.append( [sent_info_dic[sid]['text'] for sid in ss] )
    # measure similarity between system summaries and the pseudo-ref
    scores = get_sbert_score(ref_vecs, ref_tokens, summ_vecs, summ_tokens, sbert_type)

    return scores

'''


