import sys
from resources import BASE_DIR, LANGUAGE
from utils.corpus_reader import CorpusReader
from ref_free_metrics.supert import Supert
from summariser.ntd_summariser import DeepTDAgent
from summariser.ngram_vector.vector_generator import Vectoriser
from utils.evaluator import evaluate_summary_rouge, add_result
import numpy as np


class RLSummariser():
    def __init__(self,reward_func, reward_strict=5.,rl_strict=5.,train_episode=5000, base_length=200, sample_summ_num=10000):
        self.reward_func = reward_func
        self.reward_strict = reward_strict
        self.rl_strict = rl_strict
        self.train_episode = train_episode
        self.base_length = base_length
        self.sample_summ_num = sample_summ_num

    def get_sample_summaries(self, docs, summ_max_len=100):
        vec = Vectoriser(docs,summ_max_len)
        summary_list = vec.sample_random_summaries(self.sample_summ_num)
        rewards = self.reward_func(summary_list)
        assert len(summary_list) == len(rewards)
        return summary_list, rewards

    def summarise(self, docs, summ_max_len=100):
        # generate sample summaries for memory replay
        summaries, rewards = self.get_sample_summaries(docs, summ_max_len)
        vec = Vectoriser(docs,base=self.base_length)
        rl_agent = DeepTDAgent(vec, summaries, strict_para=self.rl_strict, train_round=self.train_episode)
        summary = rl_agent(rewards)
        return summary



if __name__ == '__main__':
    year = sys.argv[1]
    #year = '08' # '08' or '09'
    corpus_reader = CorpusReader(BASE_DIR)

    all_results = {}
    topic_cnt = 0

    for topic,docs,refs in corpus_reader(year):
        if '.B' in topic: continue
        print('\n=====Topic {}====='.format(topic))
        topic_cnt += 1
        supert = Supert(docs)
        rl_summariser = RLSummariser(reward_func = supert)
        summary = rl_summariser.summarise(docs, summ_max_len=100)
        print(summary)
        topic_results = {}

        for ref in refs:
            rouge_scores = evaluate_summary_rouge(summary, ref)
            add_result(topic_results, rouge_scores)
            add_result(all_results, rouge_scores)

        for metric in topic_results:
            print('{} : {:.4f}'.format(metric, np.mean(topic_results[metric])))

    print('\n=====Average results over {} topics====='.format(topic_cnt))
    for metric in all_results:
        print('{} : {:.4f}'.format(metric, np.mean(all_results[metric])))





