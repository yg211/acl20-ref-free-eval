import sys
sys.path.append('../')
import numpy as np
import os

from ref_free_metrics.sbert_score_metrics import get_rewards
from summariser.ngram_vector.vector_generator import Vectoriser
from summariser.deep_td import DeepTDAgent as RLAgent
from utils.data_reader import CorpusReader
from utils.evaluator import evaluate_summary_rouge
from resources import BASE_DIR, FEATURE_DIR


class RLSummarizer():
    def __init__(self,reward_type='top10-sbert-f1',reward_strict=5.,rl_strict=5.,train_episode=5000, base_length=200, sample_summ_num=50):
        self.reward_strict = reward_strict
        self.rl_strict = rl_strict
        self.reward_type = reward_type
        self.train_episode = train_episode
        self.base_length = base_length
        self.sample_summ_num = sample_summ_num

    def get_sample_summaries(self, docs, summ_max_len=100):
        vec = Vectoriser(docs,summ_max_len)
        summary_list = vec.sample_random_summaries(self.sample_summ_num)
        rewards = get_rewards(docs, summary_list, self.reward_type.split('-')[0])
        assert len(summary_list) == len(rewards)
        return summary_list, rewards

    def summarize(self, docs, summ_max_len=100):
        # generate sample summaries for memory replay
        summaries, rewards = self.get_sample_summaries(docs, summ_max_len)
        vec = Vectoriser(docs,base=self.base_length)
        rl_agent = RLAgent(vec, summaries, strict_para=self.rl_strict, train_round=self.train_episode)
        summary = rl_agent(rewards)
        return summary


if __name__ == '__main__':
    # read source documents
    reader = CorpusReader(BASE_DIR)
    source_docs = reader('data/topic_1/input_docs')

    # generate summaries, with summary max length 100 tokens
    rl_summarizer = RLSummarizer()
    summary = rl_summarizer.summarize(source_docs, summ_max_len=100)
    print(summary)

