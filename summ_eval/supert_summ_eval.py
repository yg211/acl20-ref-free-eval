import sys
sys.path.append('../../')
sys.path.append('../')
import numpy as np
import os
import json
from nltk.tokenize import sent_tokenize
import pickle
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, kendalltau
import math

from ref_free_metrics.supert import Supert
from utils.data_reader import CorpusReader
from utils.evaluator import evaluate_summary_rouge, add_result


def read_summ_eval_data(fpath):
    with open(fpath,'r') as ff:
        data = [json.loads(jline) for jline in ff.readlines()]

    unique_ids = list(set([entry['id'] for entry in data]))
    cleaned_data = {}
    for uid in unique_ids:
        entries = [entry for entry in data if entry['id']==uid]
        doc = entries[0]['text']
        refs = entries[0]['references']
        peers = [ee['decoded'] for ee in entries]
        scores = [ee['expert_annotations'] for ee in entries]
        cleaned_data[uid] = {'doc':doc, 'refs':refs, 'peers':peers, 'scores':scores}

    return cleaned_data


def get_human_scores(scores, wanted_score):
    assert wanted_score in ['coherence', 'fluency', 'relevance', 'consistency']
    sub_scores = []
    for entry in scores:
        ss = [ann[wanted_score] for ann in entry]
        sub_scores.append(np.mean(ss))    
    return sub_scores


def generate_supert_scores(summ_eval_data, ref_metric='top20'):
    supert_scores = {}
    for uid in tqdm(summ_eval_data):
        doc = summ_eval_data[uid]['doc']
        supert = Supert([(uid,sent_tokenize(doc))], ref_metric=ref_metric) 
        summs = summ_eval_data[uid]['peers']
        scores = supert(summs)
        supert_scores[uid] = scores
    pickle.dump(supert_scores, open('supert_summ_eval_{}.pkl'.format(ref_metric), 'wb'))

    return supert_scores


def compute_correlation(summ_eval_data, supert_scores, human_rating_type='all'):
    if human_rating_type == 'all':
        for wanted_score in ['coherence', 'fluency', 'relevance', 'consistency']:
            pccs, rhos, taus = [], [], []
            for uid in tqdm(summ_eval_data):
                human_score = get_human_scores(summ_eval_data[uid]['scores'], wanted_score)
                supert_score = supert_scores[uid]

                pcc = pearsonr(human_score, supert_score)[0]
                rho = spearmanr(human_score, supert_score)[0]
                tau = kendalltau(human_score, supert_score)[0]

                if not math.isnan(pcc): pccs.append(pcc)
                if not math.isnan(rho): rhos.append(rho)
                if not math.isnan(tau): taus.append(tau)

            print('\n===== Human Score Type: {} ====='.format(wanted_score))
            print('pcc {}, rho {}, tau {}'.format(np.mean(pccs), np.mean(rhos), np.mean(taus)))

    else:
        wanted_score = human_rating_type
        assert wanted_score in ['coherence', 'fluency', 'relevance', 'consistency']
        pccs, rhos, taus = [], [], []
        for uid in tqdm(summ_eval_data):
            human_score = get_human_scores(summ_eval_data[uid]['scores'], wanted_score)
            supert_score = supert_scores[uid]

            pcc = pearsonr(human_score, supert_score)[0]
            rho = spearmanr(human_score, supert_score)[0]
            tau = kendalltau(human_score, supert_score)[0]

            if not math.isnan(pcc): pccs.append(pcc)
            if not math.isnan(rho): rhos.append(rho)
            if not math.isnan(tau): taus.append(tau)

        print('\n===== Human Score Type: {} ====='.format(wanted_score))
        print('pcc {}, rho {}, tau {}'.format(np.mean(pccs), np.mean(rhos), np.mean(taus)))


if __name__ == '__main__':

    summ_eval_data = read_summ_eval_data('./model_annotations.aligned.paired.jsonl')

    if len(sys.argv) < 2:
        supert_scores = generate_supert_scores(summ_eval_data, 'top10')
    else:
        saved_score_path = sys.argv[1]
        supert_scores = pickle.load(open(saved_score_path,'rb'))

    compute_correlation(summ_eval_data, supert_scores, 'all')


