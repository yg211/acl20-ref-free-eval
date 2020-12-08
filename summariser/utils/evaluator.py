from sklearn.metrics import mean_squared_error
import scipy.stats as stats
from summariser.utils.rank_metrics import ndcg_at_k
from summariser.utils.misc import *
from summariser.rouge.rouge import Rouge
from summariser.utils.misc import jsd
from resources import *
from collections import OrderedDict

def evaluateReward(learnt_values, ref_values):
    assert(len(learnt_values) == len(ref_values))
    metrics_dic = OrderedDict()

    ### compute Kendall's tau, Spearman's rho and Pearson correlation coefficient
    tau, p_tau = stats.kendalltau(learnt_values, ref_values)
    rho, p_rho = stats.spearmanr(learnt_values, ref_values)
    pcc, p_pcc = stats.pearsonr(learnt_values, ref_values)
    metrics_dic['tau'] = tau
    metrics_dic['rho'] = rho
    metrics_dic['pcc'] = pcc
    metrics_dic['p_tau'] = p_tau
    metrics_dic['p_rho'] = p_rho
    metrics_dic['p_pcc'] = p_pcc

    ### compute nDCG
    # sorted_list = sorted(learnt_values,reverse=True)
    # ll = [ref_values[learnt_values.index(ele)] for ele in sorted_list]

    # ndcg = ndcg_at_k(ll,10)
    # metrics_dic['ndcg_at_10'] = ndcg
    '''
    ndcg = ndcg_at_k(ll,int(0.05*len(ll)))
    metrics_dic['ndcg_at_5%'] = ndcg
    ndcg = ndcg_at_k(ll,int(0.1*len(ll)))
    metrics_dic['ndcg_at_10%'] = ndcg
    ndcg = ndcg_at_k(ll,int(0.2*len(ll)))
    metrics_dic['ndcg_at_20%'] = ndcg
    ndcg = ndcg_at_k(ll,int(0.5*len(ll)))
    metrics_dic['ndcg_at_50%'] = ndcg
    ndcg = ndcg_at_k(ll,len(ll))
    metrics_dic['ndcg_at_all'] = ndcg
    '''

    return metrics_dic


def evaluateSummary(cand,model,len=100):
    rouge_scorer = Rouge(ROUGE_DIR,BASE_DIR,True)
    r1, r2, rl, rsu4 = rouge_scorer(cand,[model],len)
    rouge_scorer.clean()
    dic = OrderedDict()
    dic['ROUGE-1'] = r1
    dic['ROUGE-2'] = r2
    dic['ROUGE-L'] = rl
    dic['ROUGE-SU4'] = rsu4
    dic['WeightedSum'] = 3.33*(r1/.47 + r2/.212 + rsu4/.185)
    return dic

def getMetricCorrelation(metric_scores,results):
    correlations = {}

    for metric in metric_scores:
        correlations[metric] = stats.pearsonr(metric_scores[metric],results)[0]

    return correlations

def estimateTemperature(gaps,ratios):
    initial_t = 3.
    step = 0.001
    valid_num = 0
    target = 0.
    for rr in ratios:
        if not np.isnan(rr):
            valid_num += 1
            target += rr
    target /= valid_num
    if target < 0.5:
        return None, float('nan')
    round = 0
    direction = 0
    temp = initial_t

    while True:
        round += 1
        #print('\n---round {}, temperature {}---'.format(round,temp))
        prob_list = []
        for gap in gaps:
            prob_list.append(sigmoid(gap,temp))
        predict = np.mean(prob_list)
        #print('predict {}, target {}'.format(predict,target))

        if np.abs(predict-target) < 0.0001:
            break

        if predict < target:
            if direction == 1:
                break
            else:
                temp -= step
                direction = -1
        else:
            if direction == -1:
                break
            else:
                temp += step
                direction = 1

    return prob_list, temp


