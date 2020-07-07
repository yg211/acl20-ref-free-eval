import sys
from resources import BASE_DIR, LANGUAGE
from utils.corpus_reader import CorpusReader
from summariser.genetic_summariser import GeneticSummariser
from ref_free_metrics.supert import Supert
from utils.evaluator import evaluate_summary_rouge, add_result
import numpy as np

if __name__ == '__main__':
    year = sys.argv[1] # 08 or 09
    corpus_reader = CorpusReader(BASE_DIR)

    all_results = {}
    topic_cnt = 0

    for topic,docs,refs in corpus_reader(year):
        if '.B' in topic: continue
        print('\n=====Topic {}====='.format(topic))
        topic_cnt += 1
        supert = Supert(docs)
        summariser = GeneticSummariser(fitness_func=supert, max_sum_len=100)
        summary = summariser.summarise(docs)
        print(summary)
        topic_results = {}

        for ref in refs:
            rouge_scores = evaluate_summary_rouge(summary, ref)
            add_result(topic_results, rouge_scores)
            add_result(all_results, rouge_scores)

        for metric in topic_results:
            print('{} : {:.5f}'.format(metric, np.mean(topic_results[metric])))

    print('\n=====Average results over {} topics====='.format(topic_cnt))
    for metric in all_results:
        print('{} : {:.5f}'.format(metric, np.mean(all_results[metric])))




        


