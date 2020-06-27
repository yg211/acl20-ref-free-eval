import numpy as np
from nltk.tokenize import word_tokenize
import random
from tqdm import tqdm
import torch.nn as nn
import torch
import copy
import os

from ref_free_metrics.supert import Supert
from utils.data_reader import CorpusReader
from sentence_transformers import SentenceTransformer
from utils.evaluator import evaluate_summary_rouge, add_result


class GeneticSearcher():
    def __init__(self, fitness_func, max_sum_len=100, max_round=10, max_pop_size=500, temperature=0.2, mutation_rate=0.2):
        self.max_sum_len = max_sum_len
        self.temperature = temperature
        self.mutation_rate = mutation_rate
        self.max_round = max_round
        self.max_pop_size = max_pop_size

        self.fitness_func = fitness_func

    def load_data(self, docs):
        self.sentences = []
        for doc_id, doc_sents in docs:
            for si, sent in enumerate(doc_sents):
                entry = {'text': sent, 'doc_id': doc_id, 'sent_id':si, 'len':len(word_tokenize(sent))}
                self.sentences.append(entry)

    def get_avai_sents(self, avai_list, selected_sent_ids, draft_len):
        avai_ids = []
        for i in avai_list:
            if i in selected_sent_ids: continue
            if draft_len + self.sentences[i]['len'] < (self.max_sum_len*1.2):
                avai_ids += [i]
        return avai_ids


    def init_pool(self, pop_size):
        pool = []
        while len(pool) < pop_size:
            entry = []
            draft_len = 0
            avai_sent = self.get_avai_sents([i for i in range(len(self.sentences))], entry, draft_len)
            while len(avai_sent) > 0:
                idx = np.random.choice(avai_sent)
                if idx not in entry:
                    entry += [idx]
                draft_len += self.sentences[idx]['len']
                if draft_len > self.max_sum_len: break
                avai_sent = self.get_avai_sents(avai_sent, entry,draft_len)
            pool.append(entry)
        return pool    
            
    def reduce(self, new_pool, pool, new_scores, scores):
        new_pool += pool
        new_scores += scores
        threshold = np.median(new_scores)
        wanted_idx = [i for i,s in enumerate(new_scores) if s>=threshold]
        new_pool = [new_pool[i] for i in range(len(new_pool)) if i in wanted_idx]
        new_scores = [new_scores[i] for i in range(len(new_scores)) if i in wanted_idx]
        return new_pool, new_scores

    def sample_parent(self, acc_values):
        rnd = random.random()
        for i in range(len(acc_values)-1):
            if rnd >= acc_values[i] and rnd < acc_values[i+1]:
                return i


    def cross_over(self, pool, scores):
        soft_max_values = nn.Softmax(dim=0)(torch.FloatTensor(scores)/self.temperature)
        acc_values = [0.]
        for sv in soft_max_values: acc_values += [acc_values[-1]+float(sv)]
        new_pool = []
        selected_parents = []
        while len(new_pool) < len(pool):
            f = self.sample_parent(acc_values)
            m = self.sample_parent(acc_values)
            selected_parents += [f,m]
            while m == f: m = self.sample_parent(acc_values)
            min_len = min( len(pool[f]),len(pool[m]) )
            if min_len <= 1: continue
            cut_p = random.randint(1,min_len-1)
            new_f = pool[f][:cut_p] + pool[m][cut_p:]
            new_m = pool[m][:cut_p] + pool[f][cut_p:]
            new_pool += [new_f, new_m]
        new_scores = self.fitness_func(new_pool)
        #print(np.unique(selected_parents, return_counts=True))
        return new_pool, new_scores
    
    def make_new_summary(self, summary):
        new_summary = copy.deepcopy(summary)
        break_points = []
        summ_len = np.sum([self.sentences[i]['len'] for i in new_summary])
        # remove some sentences
        while summ_len > self.max_sum_len:
            sent_num = len(new_summary)
            p = random.randint(0,sent_num-1)
            summ_len -= self.sentences[new_summary[p]]['len']
            del new_summary[p]
        # add new sentences
        avai_sents = [i for i in range(len(self.sentences))] 
        avai_sents = self.get_avai_sents(avai_sents, new_summary, summ_len)
        while len(avai_sents) > 0:
           new_summary += [np.random.choice(avai_sents)]
           summ_len += self.sentences[new_summary[-1]]['len']
           if summ_len > self.max_sum_len: break
           avai_sents = self.get_avai_sents(avai_sents, new_summary, summ_len)
        # compute score
        new_score = self.fitness_func([new_summary])[0]
        return new_summary, new_score


    def mutate(self, pool, scores):
        idx = []
        new_scores = []
        for i in range(len(pool)):
            if random.random() > self.mutation_rate: continue
            new_summary, new_score = self.make_new_summary(pool[i])
            pool[i] = new_summary
            scores[i] = new_score
        

    def round_search(self, pool, scores):
        new_pool, new_scores = self.cross_over(pool, scores)
        self.mutate(new_pool, new_scores)
        new_pool, new_scores = self.reduce(new_pool, pool, new_scores, scores)

        return new_pool, new_scores


    def summarize(self, docs):
        self.load_data(docs) 
        pool = self.init_pool(self.max_pop_size)
        scores = self.fitness_func(pool)

        for i in tqdm(range(self.max_round)):
            pool, scores = self.round_search(pool, scores)
            print('round {}, max fitness {:.3f}, median fitness {:.3f}'.format(i, np.max(scores), np.median(scores)))

        summ_idx = pool[np.argmax(scores)]
        summary = ' '.join([self.sentences[si]['text'] for si in summ_idx])
        return summary

 
if __name__ == '__main__':
    # read source documents
    reader = CorpusReader('data/topic_1')
    source_docs = reader()

    # generate summaries using genetic algorithm, with supert as fitness function
    supert = Supert(source_docs)
    summarizer = GeneticSearcher(fitness_func=supert, max_sum_len=100)
    summary = summarizer.summarize(source_docs)
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
       




