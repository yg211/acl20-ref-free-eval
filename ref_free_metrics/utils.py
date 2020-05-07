import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize


def get_human_score(topic, summ_name, human):
    block = summ_name.split('-')[1].split('.')[0]
    id = summ_name.split('.')[-1]
    key = 'topic{}-{}_sum{}'.format(topic.split('.')[0],block,id)
    if key not in human: return None
    else: return human[key]

def get_idf_weights(ref_vecs):
    sim_matrix = cosine_similarity(ref_vecs,ref_vecs)
    #dfs = [np.sort(sim_matrix[i])[-2] for i in range(len(ref_vecs))]
    dfs = [np.sum(sim_matrix[i])-1. for i in range(len(ref_vecs))]
    dfs = [1.*d/(len(ref_vecs)-1) for d in dfs]
    dfs = [(d+1.)/2. for d in dfs]
    idf = [-1.*np.log(df) for df in dfs]
    return idf


def get_ref_vecs(doc_sent_vecs, doc_sent_weights, info_dic):
    ref_dic = {}
    docs = set([info_dic[k]['doc'] for k in info_dic])
    for dd in docs:
        ref_dic[dd] = [i for i in range(len(doc_sent_weights)) if doc_sent_weights[i]>=0.1 and info_dic[i]['doc']==dd]
    vecs = []
    for dd in ref_dic:
        allv = np.array(doc_sent_vecs)[ref_dic[dd]]
        meanv = np.mean(allv,axis=0)
        vecs.append(meanv)
    return vecs


def get_sim_metric(summ_vec_list, doc_sent_vecs, doc_sent_weights, info_dic, method='cos'):
    #print('weights', doc_sent_weights)
    # get the avg doc vec, then cosine
    if method == 'cos':
        summ_vec = np.mean(np.array(summ_vec_list),axis=0)
        dvec = np.matmul(np.array(doc_sent_weights).reshape(1,-1),  np.array(doc_sent_vecs))
        return cosine_similarity(dvec,summ_vec.reshape(1,-1))[0][0]
        # below: good performance with true_ref, poorer performance with other pseduo-refs
        #ref_vecs = get_ref_vecs(doc_sent_vecs, doc_sent_weights, info_dic)
        #sims = cosine_similarity(np.array(ref_vecs), np.array(summ_vec).reshape(1,-1))
        #return np.mean(sims)
    # bert-score, quicker to run and gives similar performance to mover-bert-score
    else:
        ref_vecs = [doc_sent_vecs[i] for i in range(len(doc_sent_weights)) if doc_sent_weights[i]>0.1]
        #ref_vecs = get_ref_vecs(doc_sent_vecs, doc_sent_weights, info_dic)
        weights = [doc_sent_weights[i] for i in range(len(doc_sent_weights)) if doc_sent_weights[i]>0.1]
        idf_weights = get_idf_weights(ref_vecs)
        sim_matrix = cosine_similarity(np.array(ref_vecs),np.array(summ_vec_list))
        recall = np.mean(np.max(sim_matrix,axis=1))
        idf_recall = np.dot(np.max(sim_matrix,axis=1),idf_weights)/np.sum(idf_weights)
        precision = np.mean(np.max(sim_matrix,axis=0))
        if recall+precision == 0:
            f1 = None
        else:
            f1 = 2.*recall*precision/(recall+precision)
            idf_f1 = 2.*idf_recall*precision/(idf_recall+precision)
        if method.lower().startswith('f'): return f1
        elif method.lower().startswith('r'): return recall
        elif method.lower().startswith('p'): return precision
        elif method.lower().startswith('idf'):
            if 'recall' in method: return idf_recall
            elif 'f1' in method: return idf_f1
            else: return None
        elif method.lower().startswith('w'): return np.dot(np.array(np.max(sim_matrix,axis=1)),np.array(weights))/np.sum(weights)
        else: return None


def parse_docs(docs,bert_model):
    all_sents = []
    sent_index = {}
    cnt = 0
    for dd in docs:
        dname = dd[0].split('/')[-1]
        doc_len = len(dd[1])
        for i, sent in enumerate(dd[1]):
            sent_index[cnt] = {'doc': dname, 'text': sent, 'inside_doc_idx': i, 'doc_len': doc_len,
                               'inside_doc_position_ration': i * 1. / doc_len}
            cnt += 1
            all_sents.append(sent)
    all_vecs = None
    if bert_model is not None:
        all_vecs = bert_model.encode(all_sents)
    return sent_index, all_vecs #, all_sents


def parse_refs(refs,model):
    all_sents = []
    sent_index = {}
    cnt = 0
    for i,rr in enumerate(refs):
        if len(rr[1]) == 1: # TAC09, one piece of text
            ref_sents = sent_tokenize(rr[1][0])
        else: # TAC08, a list of sentences
            ref_sents = rr[1]
        ref_name = 'ref{}'.format(i)
        for j, sent in enumerate(ref_sents):
            sent_index[cnt] = {'doc': ref_name, 'text': sent, 'inside_doc_idx': j, 'doc_len': len(ref_sents),
                               'inside_doc_position_ration': j * 1. / len(ref_sents)}
            cnt += 1
            all_sents.append(sent)
    all_vecs = None
    if model is not None:
        all_vecs = model.encode(all_sents)
    return sent_index, all_vecs

