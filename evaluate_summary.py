import sys
sys.path.append('../')
import numpy as np
import os

from ref_free_metrics.sbert_score_metrics import get_sbert_score_metrics
from utils.data_reader import CorpusReader
from utils.evaluator import evaluate_summary_rouge, add_result


if __name__ == '__main__':
    pseudo_ref = 'top15' # pseudo-ref strategy
    similarity_measurement = 'sbert_score' # measure similarity between summary and pseudo-ref

    # read source documents
    reader = CorpusReader('data/topic_1')
    source_docs = reader()
    summaries = reader.readSummaries()

    # get unsupervised metrics for the summaries
    if similarity_measurement.startswith('sbert_score'):
        scores = get_sbert_score_metrics(source_docs, summaries, pseudo_ref,mute=False)

    # compare the summaries against golden refs using ROUGE
    refs = reader.readReferences() # make sure you have put the references in data/topic_1/references
    summ_rouge_scores = []
    for summ in summaries:
        rouge_scores = {}
        for ref in refs:
            rs = evaluate_summary_rouge(summ, ref)
            add_result(rouge_scores, rs)
        summ_rouge_scores.append(rouge_scores)

    mm = 'ROUGE-1'
    rouge_scores = []
    for rs in summ_rouge_scores:
        rouge_scores.append( np.mean(rs[mm]) )

    print('unsupervised metrics\n', scores)
    print('reference-based',mm,'\n',rouge_scores)




