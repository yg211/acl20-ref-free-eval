import sys
sys.path.append('../')
import numpy as np
import os

from ref_free_metrics.supert import Supert
from summariser.ngram_vector.vector_generator import Vectoriser
from summariser.deep_td import DeepTDAgent as RLAgent
from utils.data_reader import CorpusReader
from utils.evaluator import evaluate_summary_rouge, add_result


class RLSummarizer():
    def __init__(self,reward_func, reward_strict=5.,rl_strict=5.,train_episode=5000, base_length=200, sample_summ_num=5000, gpu=True):
        self.reward_func = reward_func
        self.reward_strict = reward_strict
        self.rl_strict = rl_strict
        self.train_episode = train_episode
        self.base_length = base_length
        self.sample_summ_num = sample_summ_num
        self.gpu = gpu

    def get_sample_summaries(self, docs, summ_max_len=100):
        vec = Vectoriser(docs,summ_max_len)
        summary_list = vec.sample_random_summaries(self.sample_summ_num)
        rewards = self.reward_func(summary_list)
        assert len(summary_list) == len(rewards)
        return summary_list, rewards

    def summarize(self, docs, summ_max_len=100):
        # generate sample summaries for memory replay
        summaries, rewards = self.get_sample_summaries(docs, summ_max_len)
        vec = Vectoriser(docs, sum_len=summ_max_len, base=self.base_length)
        rl_agent = RLAgent(vec, summaries, strict_para=self.rl_strict, train_round=self.train_episode, gpu=self.gpu)
        summary = rl_agent(rewards)
        return summary


if __name__ == '__main__':
    # read source documents
    reader = CorpusReader('data/topic_1')
    source_docs = reader()

    # generate summaries, with summary max length 100 tokens
    supert = Supert(source_docs)
    rl_summarizer = RLSummarizer(reward_func = supert)
    summary = rl_summarizer.summarize(source_docs, summ_max_len=50)
    print('\n=====Generated Summary=====')
    print(summary)

    # (Optional) Evaluate the quality of the summary using ROUGE metrics
    if os.path.isdir('./rouge/ROUGE-RELEASE-1.5.5'):
        refs = reader.readReferences() # make sure you have put the references in data/topic_1/references
        avg_rouge_score = {}
        for ref in refs:
            rouge_scores = evaluate_summary_rouge(summary, ref)
            add_result(avg_rouge_score, rouge_scores)
        print('\n=====ROUGE scores against {} references====='.format(len(refs)))
        for metric in avg_rouge_score:
            print('{}:\t{}'.format(metric, np.mean(rouge_scores[metric])))

