import numpy as np
import random
import math
import scipy as sp

def normaliseList(ll,max_value=10.):
    minv = min(ll)
    maxv = max(ll)
    gap = maxv-minv

    new_ll = [(x-minv)*max_value/gap for x in ll]

    return new_ll

def sigmoid(x,temp=1.):
    return 1.0/(1.+math.exp(-x/temp))


def softmax_sample(value_list,strict,softmax_list=[],return_softmax_list=False):
    if len(softmax_list) == 0:
        slist = getSoftmaxList(value_list,strict)
    else:
        slist = softmax_list

    pointer = random.random()*sum(softmax_list)
    tier = 0
    idx = 0

    rtn_idx = -1
    for value in slist:
        if pointer >= tier and pointer < tier+value:
            rtn_idx = idx
            break
        else:
            tier += value
            idx += 1

    if return_softmax_list:
        return rtn_idx,slist
    else:
        return rtn_idx


def getSoftmaxList(value_list, strict):
    softmax_list = []
    for value in value_list:
        softmax_list.append(np.exp(value/strict))
    return softmax_list

def getSoftmaxProb(value_list, strict):
    slist = getSoftmaxList(value_list,strict)
    return [xx/np.sum(slist) for xx in slist]


def cosine(vec1,vec2):
    return np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))


def jsd(p, q, wanted='js', base=np.e):
    '''
        Implementation of pairwise `jsd` based on
        https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    '''
    ## convert to np.array
    p, q = np.asarray(p), np.asarray(q)
    ## normalize p, q to probabilities
    if p.sum() == 0 or q.sum() == 0:
        return -1.

    p, q = p/p.sum(), q/q.sum()
    m = 1./2*(p + q)

    if wanted == 'kl1': return sp.stats.entropy(p,q,base=base)
    elif wanted == 'kl2': return sp.stats.entropy(q,p,base=base)
    else:
        assert wanted == 'js'
        return sp.stats.entropy(p,m,base=base)/2.+ sp.stats.entropy(q,m,base=base)/2.


def getNormMeanFromLogMean(log_mean,log_dev):
    ### from the mean/dev of log-normal to obtain mean/dev for normal distribution
    ### see wikipedia of log normal distribution
    norm_mean = np.log(log_mean/math.sqrt(1+math.pow(log_dev,2)/math.pow(log_mean,2)))
    norm_dev = np.log( 1+math.pow(log_dev,2)/math.pow(log_mean,2) )

    return norm_mean, math.sqrt(norm_dev)

def bellCurvise(js_list,mean=5.,dev=2.,norm=False,log=True):
    ### note that the smaller the js, the better the summary
    sorted_js = sorted(js_list)
    if log:
        norm_mean, norm_dev = getNormMeanFromLogMean(mean,dev)
        norm_values = list(np.random.lognormal(norm_mean,norm_dev,len(js_list)))
    else:
        norm_values = list(np.random.normal(mean,dev,len(js_list)))
    norm_values = sorted(norm_values,reverse=True)
    rewards = []

    for js in js_list:
        rewards.append(norm_values[sorted_js.index(js)])

    if norm:
        return normaliseList(rewards)
    else:
        return rewards


def aggregateScores(scores_dic):
    score_matrix = []

    for name in scores_dic:
        score_matrix.append(scores_dic[name])

    return np.array(score_matrix).mean(0)

def getRankBasedScores(scores,normalise=True):
    rewards = scores[:]
    sr = sorted(rewards)
    for i in range(len(sr)-1):
        if sr[i] == sr[i+1]:
            dec_value = random.random()*1e-5
            sr[i] -= dec_value
            rewards[rewards.index(sr[i+1])] -= dec_value
    sr = sorted(sr)

    rank_rewards = []
    for rr in rewards:
        rank_rewards.append(sr.index(rr))

    if normalise:
        return np.array(rank_rewards)*10./len(rank_rewards)
    else:
        return np.array(rank_rewards)




