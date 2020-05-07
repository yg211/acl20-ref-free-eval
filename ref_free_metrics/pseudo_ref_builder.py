from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np
from tqdm import tqdm
from sklearn.cluster import AffinityPropagation


def get_doc_simtop(sim_matrix, max_sim_value):
    nn = sim_matrix.shape[0]
    for i in range(1,sim_matrix.shape[0]):
        if np.max(sim_matrix[i][:i])>max_sim_value: 
            nn = i
            break
    return nn


def get_top_sim_weights(sent_info, full_vec_list, max_sim_value):
    doc_names = set([sent_info[k]['doc'] for k in sent_info])
    weights = [0.]*len(sent_info)
    for dn in doc_names:
        doc_idx = [k for k in sent_info if sent_info[k]['doc']==dn]
        sim_matrix = cosine_similarity(np.array(full_vec_list)[doc_idx], np.array(full_vec_list)[doc_idx])
        nn = get_doc_simtop(sim_matrix, max_sim_value)
        for i in range(np.min(doc_idx),np.min(doc_idx)+nn): weights[i] = 1.
    return weights


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
    subgraph = [gg.subgraph(c) for c in nx.connected_components(gg)]
    subgraph_nodes = [list(sg._node.keys()) for sg in subgraph]
    return list(subgraph_nodes)


def get_other_weights(full_vec_list, sent_index, weights, thres):
    similarity_matrix = cosine_similarity(full_vec_list, full_vec_list)
    subgraphs = get_subgraph(similarity_matrix, thres)
    for sg in subgraphs:
        if any(weights[n]>=0.9 for n in sg): continue #ignore the subgraph similar to a top sentence
        if len(set([sent_index[n]['doc'] for n in sg])) < 2: continue #must appear in multiple documents
        for n in sg: weights[n]=1./len(sg)
        #print(sg,'added to weights')


def graph_centrality_weight(similarity_matrix):
    weights_list = [np.sum(similarity_matrix[i])-1. for i in range(similarity_matrix.shape[0])]
    return weights_list


def graph_weights(full_vec_list):
    similarity_matrix = cosine_similarity(full_vec_list, full_vec_list)
    weights_list = graph_centrality_weight(similarity_matrix)
    return weights_list


def get_indep_graph_weights(sent_info_dic, sent_vecs, num, top_n, extra_ratio):
    doc_names = set([sent_info_dic[key]['doc'] for key in sent_info_dic])
    wanted_id = []
    for dname in doc_names:
        ids = np.array([key for key in sent_info_dic if sent_info_dic[key]['doc']==dname])
        doc_weights = np.array(graph_weights(np.array(sent_vecs)[ids]))
        if top_n is not None: 
            for j in range(top_n): 
                if j>=len(doc_weights): break
                doc_weights[j] *= extra_ratio
        wanted_id.extend(list(ids[doc_weights.argsort()[-num:]]))
    weights = [0.]*len(sent_vecs)
    for ii in wanted_id: weights[ii] = 1.
    return weights


def get_global_graph_weights(sent_info_dic, sent_vecs, num, top_n, extra_ratio):
    raw_weights = graph_weights(sent_vecs)
    if top_n is not None:
        top_ids = [i for i in sent_info_dic if sent_info_dic[i]['inside_doc_idx']<top_n]
        adjusted_weights = [w*extra_ratio if j in top_ids else w for j,w in enumerate(raw_weights) ]
    else:
        adjusted_weights = raw_weights
    wanted_id = np.array(adjusted_weights).argsort()[-num:]
    weights = [0.] * len(sent_vecs)
    for ii in wanted_id: weights[ii] = 1.
    return weights


def get_indep_cluster_weights(sent_info_dic, sent_vecs):
    doc_names = set([sent_info_dic[key]['doc'] for key in sent_info_dic])
    sums = [np.sum(sv) for sv in sent_vecs]
    wanted_ids = []
    for dname in doc_names:
        sids = np.array([key for key in sent_info_dic if sent_info_dic[key]['doc']==dname])
        clustering = AffinityPropagation().fit(np.array(sent_vecs)[sids])
        centers = clustering.cluster_centers_
        for cc in centers: wanted_ids.append(sums.index(np.sum(cc)))
    print('indep cluster, pseudo-ref sent num', len(wanted_ids))
    weights = [1. if i in wanted_ids else 0. for i in range(len(sent_vecs))]
    return weights


def get_global_cluster_weights(sent_vecs):
    clustering = AffinityPropagation().fit(sent_vecs)
    centers = clustering.cluster_centers_
    print('global cluster, pseudo-ref sent num', len(centers))
    sums = [np.sum(sv) for sv in sent_vecs]
    ids = []
    for cc in centers: ids.append(sums.index(np.sum(cc)))
    assert len(ids) == len(centers)
    weights = [1. if i in ids else 0. for i in range(len(sent_vecs))]
    return weights

'''
def give_top_extra_weights(weights, sent_info_dic, top_n, extra_ratio):
    for ii in sent_info_dic:
        if sent_info_dic[ii]['inside_doc_idx']<top_n: weights[ii]*extra_ratio
'''





