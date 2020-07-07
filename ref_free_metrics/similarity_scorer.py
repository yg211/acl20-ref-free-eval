import sys
sys.path.append('../')
import os
from tqdm import tqdm
import numpy as np
import random

from pseudo_ref_builder import *
from resources import BASE_DIR
from utils import parse_refs, parse_docs, get_sim_metric
from summariser.utils.misc import normaliseList


def get_weights(sent_info_dic, sent_vecs, metric):
    if metric == 'full_doc':
        weights = [1.]*len(sent_info_dic)
    elif metric.startswith('random'):
        if '_' in metric:
            ref_length = int(metric.split('_')[1])
        else:
            ref_length = 10 # by default we randomly select 10 sents from each doc as the pseudo-ref
        weights = []
        for key in sent_info_dic:
            ratio = ref_length*1./sent_info_dic[key]['doc_len']
            if random.random() < ratio:
                weights.append(1.)
            else:
                weights.append(0.)
    elif metric.startswith('indep_graph') or metric.startswith('global_graph'):
        eles = metric.split('_')
        num = int(eles[2][3:])
        if 'extra' in metric:
            assert len(eles) == 5
            top_n = int(eles[3][5:])
            extra_amp = float(eles[-1])
        else:
            extra_amp = None
            top_n = None
        if 'indep' in metric:
            weights = get_indep_graph_weights(sent_info_dic, sent_vecs, num, top_n, extra_amp)
        else:
            weights = get_global_graph_weights(sent_info_dic, sent_vecs, num, top_n, extra_amp)
    elif metric.startswith('global_cluster'):
        weights = get_global_cluster_weights(sent_vecs)
    elif metric.startswith('indep_cluster'):
        weights = get_indep_cluster_weights(sent_info_dic, sent_vecs)
    elif metric.startswith('simmax'):
        simmax = float(metric.split('_')[1])
        weights = get_top_sim_weights(sent_info_dic, sent_vecs,simmax)
    else:
        assert metric.startswith('top')
        if '_' in metric:
            topn = int(metric.split('_')[0][3:])
            thres = float(metric.split('_')[1])
        else:
            topn = int(metric[3:])
            thres = 100.
        weights = get_top_weights(sent_info_dic, topn)
        if thres > 0:
            get_other_weights(sent_vecs, sent_info_dic, weights, thres)
    return weights


def get_similarity_score(bert_model, summ_list, sents_weights, sent_vecs, sent_info_dic, sim_metric):
    sim_scores = []
    none_flags = []
    for ss in tqdm(summ_list):
        if len(ss) == 0:
            none_flags.append(1)
            continue
        none_flags.append(0)
        summ_vecs = bert_model.encode(ss)
        sims = get_sim_metric(summ_vecs, sent_vecs, sents_weights, sent_info_dic, sim_metric)
        sim_scores.append(sims)
    sim_scores = normaliseList(sim_scores)
    scores = []
    cnt = 0
    for i in range(len(none_flags)):
        if none_flags[i] == 1:
            scores.append(None)
        else:
            scores.append(sim_scores[cnt])
            cnt += 1

    return scores


def get_entail_label(model, doc_sents_dic, doc_sents_weights, threshold, ss):
    if set(doc_sents_weights) != set([0,1]):
        scu_idx = [i for i in range(len(doc_sents_weights)) if doc_sents_weights[i]>=threshold]
    else:
        scu_idx = [i for i in range(len(doc_sents_weights)) if doc_sents_weights[i]==1]

    sent_pairs = [(doc_sents_dic[ii]['text'],ss) for ii in scu_idx]
    labels, probs = model(sent_pairs)
    entail_num = len([ll for ll in labels if 'entail' in ll])
    contra_num = len([ll for ll in labels if 'contradiction' in ll])
    return entail_num, contra_num #, len(labels)


def get_nli_metric(model, summ_sents, doc_sents_dic,doc_sents_weights,threshold=0.1):
    entail_scores = []
    for ss in summ_sents:
        entail, contr = get_entail_label(model, doc_sents_dic, doc_sents_weights, threshold, ss)
        entail_scores.append(entail) #-contr)
    scores = []
    for es in entail_scores:
        if es >= 1: scores.append(1)
        else: scores.append(0)
    return np.mean(scores)


def get_nli_score(nli_model, summ_list, sents_weights, sent_info_dic):
    nli_scores = []
    none_flags = []
    for ss in tqdm(summ_list):
        if len(ss) == 0:
            none_flags.append(1)
            continue
        none_flags.append(0)
        entail = get_nli_metric(nli_model, ss, sent_info_dic, sents_weights)
        nli_scores.append(entail)
    nli_scores = normaliseList(nli_scores)
    scores = []
    cnt = 0
    for i in range(len(none_flags)):
        if none_flags[i] == 1:
            scores.append(None)
        else:
            scores.append(nli_scores[cnt])
            cnt += 1

    return scores


def parse_documents(docs, bert_model, ref_metric, debug=False):
    if ref_metric == 'true_ref':
        sent_info_dic, sent_vecs = parse_refs(docs,bert_model)
        sents_weights = [1.]*len(sent_info_dic)
    else:
        sent_info_dic, sent_vecs = parse_docs(docs,bert_model)
        sents_weights = get_weights(sent_info_dic, sent_vecs, ref_metric)
    if debug:
        pseudo_ref = [sent_info_dic[k]['text'] for k in sent_info_dic if sents_weights[k]>0.1]
        print('=====pseudo ref=====')
        print('\n'.join(pseudo_ref))
    return sent_info_dic, sent_vecs, sents_weights


def get_scores(docs, summs, bert_model, ref_metric, sim_metric, debug=False):
    sent_info_dic, sent_vecs, sents_weights = parse_documents(docs,bert_model,ref_metric,debug)
    pss = get_similarity_score(bert_model, summs, sents_weights, sent_vecs, sent_info_dic, sim_metric)
    return pss



