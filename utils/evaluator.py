from rouge.rouge import Rouge
from resources import *
from collections import OrderedDict

def add_result(all_dic,result):
    for metric in result:
        if metric in all_dic:
            all_dic[metric].append(result[metric])
        else:
            all_dic[metric] = [result[metric]]


def evaluate_summary_rouge(cand,model,max_sum_len=100):
    rouge_scorer = Rouge(ROUGE_DIR,BASE_DIR,True)
    r1, r2, rl, rsu4 = rouge_scorer(cand,[model],max_sum_len)
    rouge_scorer.clean()
    dic = OrderedDict()
    dic['ROUGE-1'] = r1
    dic['ROUGE-2'] = r2
    dic['ROUGE-L'] = rl
    dic['ROUGE-SU4'] = rsu4
    return dic

