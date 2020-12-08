import sys
sys.path.append('../..')

from resources import BASE_DIR,FEATURE_DIR
from summariser.data_processor.corpus_reader import CorpusReader
from summariser.data_processor.sys_summ_reader import PeerSummaryReader
from summariser.utils.reader import readSummaries
from summariser.utils.misc import normaliseList
from nli.bert_nli import BertNLIModel

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import networkx as nx


def get_top_weights(sent_index, topn):
    weights = []
    for i in range(len(sent_index)):
        if sent_index[i]['inside_doc_idx'] < topn:
            weights.append(1.)
        else:
            weights.append(0.)
    return weights


def get_subgraph(sim_matrix, threshold):
    gg = nx.Graph()
    for i in range(0,sim_matrix.shape[0]-1):
        for j in range(i+1,sim_matrix.shape[0]):
            if sim_matrix[i][j] >= threshold:
                gg.add_node(i)
                gg.add_node(j)
                gg.add_edge(i,j)
    subgraph = list(nx.connected_component_subgraphs(gg))
    subgraph_nodes = [list(sg._node.keys()) for sg in subgraph]
    return list(subgraph_nodes)


def get_other_weights(full_vec_list, sent_index, weights, thres):
    similarity_matrix = cosine_similarity(full_vec_list, full_vec_list)
    subgraphs = get_subgraph(similarity_matrix, thres)
    '''
    top_sent_idx = [i for i in range(len(weights)) if weights[i]>0.9]
    for sg in subgraphs:
        if len(set([sent_index[n]['doc'] for n in sg])) < 2: continue #must appear in multiple documents
        for n in sg: weights[n]=1./len(sg)
    '''
    for sg in subgraphs:
        if any(weights[n]>=0.9 for n in sg): continue #ignore the subgraph similar to a top sentence
        if len(set([sent_index[n]['doc'] for n in sg])) < 2: continue #must appear in multiple documents
        for n in sg: weights[n]=1./len(sg)
        print(sg,'added to weights')


def get_sents_weights(docs, model, args, thres):
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

    if 'bert-sts' in args:
        all_vecs = model.encode(all_sents)
    elif 'tfidf' in args:
        all_vecs = model.fit_transform(all_sents).toarray()

    # give high weights to top sentences
    weights = get_top_weights(sent_index, int([a for a in args.split('-') if 'top' in a][0][3:]))
    if thres > 0:
        get_other_weights(all_vecs, sent_index, weights, thres)

    return weights, sent_index, all_vecs


def get_sim_metric(summ_vec_list, doc_sent_vecs, doc_sent_weights, method=1):
    #print('weights', doc_sent_weights)
    # method 1: get the avg doc vec, then cosine
    if method == 1:
        summ_vec = np.mean(np.array(summ_vec_list),axis=0)
        dvec = np.matmul(np.array(doc_sent_weights).reshape(1,-1),  np.array(doc_sent_vecs))
        return cosine_similarity(dvec,summ_vec.reshape(1,-1))[0][0]

    # method 2: cosine between each doc and the summ, then avg
    elif method == 2:
        summ_vec = np.mean(np.array(summ_vec_list),axis=0)
        sim_matrix = cosine_similarity(np.array(doc_sent_vecs),summ_vec.reshape(1,-1))
        mm = np.matmul(np.array(sim_matrix).reshape(1,-1),np.array(doc_sent_weights)).reshape(-1,1)[0][0]
        return mm

    else: 
        ref_vecs = [doc_sent_vecs[i] for i in range(len(doc_sent_weights)) if doc_sent_weights[i]>0.1]
        weights = [doc_sent_weights[i] for i in range(len(doc_sent_weights)) if doc_sent_weights[i]>0.1]
        sim_matrix = cosine_similarity(np.array(ref_vecs),np.array(summ_vec_list))
        recall = np.mean(np.max(sim_matrix,axis=1))
        precision = np.mean(np.max(sim_matrix,axis=0))
        if recall+precision == 0:
            f1 = None
        else:
            f1 = 2.*recall*precision/(recall+precision)
        if method.lower().startswith('f'): return f1
        elif method.lower().startswith('r'): return recall
        elif method.lower().startswith('p'): return precision
        elif method.lower().startswith('w'): return np.dot(np.array(np.max(sim_matrix,axis=1)),np.array(weights))/np.sum(weights)
        else: return None


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


def get_nli_metric(model, summ_sents,doc_sents_dic,doc_sents_weights,threshold=0.1):
    entail_scores = []
    for ss in summ_sents:
        entail, contr = get_entail_label(model, doc_sents_dic, doc_sents_weights, threshold, ss)
        entail_scores.append(entail) #-contr)
    scores = []
    for es in entail_scores:
        if es >= 1: scores.append(1)
        else: scores.append(0)
    return np.mean(scores)


def get_score(sent_info_dic, summ_list, sents_weights, sent_vecs, args):
    if 'nli' in args:
        nli_model = BertNLIModel(os.path.join(BASE_DIR, 'nli','model.state_dict'))

    sim_scores = []
    nli_scores = []
    for ss in tqdm(summ_list,desc='iterate over summaries'):
        summ_vecs = [sent_vecs[sid] for sid in ss]
        summ_sents = [sent_info_dic[sid]['text'] for sid in ss]

        if 'nli' in args:
            entail = get_nli_metric(nli_model,summ_sents,sent_info_dic,sents_weights)
            nli_scores.append(entail)
        else:
            sims = get_sim_metric(summ_vecs, sent_vecs, sents_weights)
            sim_scores.append(sims)

    if 'nli' in args:
        return nli_scores #normaliseList(nli_scores,1.)
    else:
        return sim_scores #normaliseList(sim_scores,1.)


if __name__ == '__main__':
    year = '08'
    scu_finder = 'bert-sts-top1-nli' #rouge, tfidf, bert-sts
    thres = -1
    print('data TAC{}, scu_finder {}, threshold {}'.format(year, scu_finder, thres))

    out_base = os.path.join(FEATURE_DIR,year)
    if not os.path.exists(out_base):
        os.makedirs(out_base)

    corpus_reader = CorpusReader(BASE_DIR)
    peer_summaries = PeerSummaryReader(BASE_DIR)(year)

    if 'bert-sts' in scu_finder:
        model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')
    elif 'tfidf' in scu_finder:
        model = TfidfVectorizer(ngram_range=(1,1),preprocessor=PorterStemmer().stem)

    sent_nums = []
    for topic,docs,models in tqdm(corpus_reader(year)):
        # encode docs
        if '.B' in topic: continue
        fpath = os.path.join(out_base, topic, 'pseudo_ref1_nli')
        if os.path.exists(fpath): continue
        print('\n=====topic {}====='.format(topic))
        sents_weights, sent_info_dic, sent_vecs = get_sents_weights(docs,model,scu_finder,thres)
        summs, ref_values_dic = readSummaries(year,topic,'rouge',sample_num=10000)
        rewards = get_score(sent_info_dic, summs, sents_weights, sent_vecs, scu_finder)

        out_str = ''
        for ii, vv in enumerate(rewards):
            out_str += '{}\t{}\n'.format(summs[ii], vv)

        if not os.path.exists(os.path.join(out_base, topic)):
            os.makedirs(os.path.join(out_base, topic))

        ff = open(fpath, 'w')
        ff.write(out_str)
        ff.close()




