
from summariser.utils.misc import jsd,normaliseList
from nltk.stem.porter import PorterStemmer
from collections import OrderedDict
from nltk.corpus import stopwords
from summariser.utils.data_helpers import sent2stokens_wostop,extract_ngrams2
from resources import LANGUAGE,FEATURE_DIR,BASE_DIR
from summariser.utils.reader import readSummaries
from summariser.data_processor.corpus_reader import CorpusReader

import numpy as np
import os

class RougeRewardGenerator:
    def __init__(self,docs,nlist=None):
        self.stopwords = set(stopwords.words(LANGUAGE))
        self.stemmer = PorterStemmer()

        self.docs = docs
        self.sentences = []
        for doc in docs:
            self.sentences.extend(doc[1])
        self.vocab = None
        if nlist is None:
            self.nlist = [1]
        else:
            self.nlist = nlist
        self.vocab, self.doc_word_distribution = self.getWordDistribution(self.sentences,self.vocab,self.nlist)


    def __call__(self, summary_list):
        rouge_list = []
        for summary_idx in summary_list:
            summary = []
            for idx in summary_idx:
                summary.append(self.sentences[idx])
            _, sum_word_distrubtion = self.getWordDistribution(summary,self.vocab,self.nlist)
            rscore = np.sum([1 for cc in sum_word_distrubtion if cc is not 0])*1./len(sum_word_distrubtion)
            rouge_list.append(rscore)

        return rouge_list

    def getWordDistribution(self,sent_list,vocab,nlist):
        if vocab is None:
            vocab_list = []
            build_vocab = True
        else:
            vocab_list = vocab
            build_vocab = False

        dic = OrderedDict((el,0) for el in vocab_list)

        for sent in sent_list:
            ngrams = []
            for n in nlist:
                if n == 1:
                    ngrams.extend(sent2stokens_wostop(sent,self.stemmer,self.stopwords,LANGUAGE))
                else:
                    ngrams.extend(extract_ngrams2([sent],self.stemmer,LANGUAGE,n))

            for ww in ngrams:
                if ww in dic:
                    dic[ww] = dic[ww]+1
                elif build_vocab:
                    dic[ww] = 1

        return list(dic.keys()), list(dic.values())


if __name__ == '__main__':
    dataset = '08' ## 08, 09
    sample_num = 10000
    out_base = os.path.join(FEATURE_DIR,dataset)
    if not os.path.exists(out_base):
        os.makedirs(out_base)

    ### read documents and ref. summaries
    reader = CorpusReader(BASE_DIR)
    data = reader(dataset)

    ### store all results
    all_test_reward_dic = OrderedDict()
    topic_cnt = 0

    summaries = []
    targets = []
    groups = []
    models_list = []
    docs_list = []
    rougev = 2

    ### read data
    for topic,docs,models in data:
        if '.B' in topic: continue
        #if not ('0828' in topic or '0829' in topic or '0830' in topic): continue

        print('read DATA {}, TOPIC {}'.format(dataset,topic))
        summs, ref_values_dic = readSummaries(dataset,topic,'rouge',sample_num)
        rouge = RougeRewardGenerator(docs,[rougev])
        rouge_values = rouge(summs)
        print('topic {}, rouge{} value num {}, mean {}, max {}, min {}'.format(topic,rougev,len(rouge_values), np.mean(rouge_values), np.max(rouge_values), np.min(rouge_values)))
        out_str = ''
        for ii,vv in enumerate(rouge_values):
            out_str += '{}\t{}\n'.format(summs[ii],vv)

        if not os.path.exists(os.path.join(out_base,topic)):
            os.makedirs(os.path.join(out_base,topic))

        fpath = os.path.join(out_base,topic,'rouge{}'.format(rougev))
        ff = open(fpath,'w')
        ff.write(out_str)
        ff.close()








