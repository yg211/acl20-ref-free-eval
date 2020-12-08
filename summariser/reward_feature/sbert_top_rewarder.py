import sys
sys.path.append('../..')

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import os

from resources import BASE_DIR,FEATURE_DIR
from summariser.data_processor.corpus_reader import CorpusReader
from summariser.data_processor.sys_summ_reader import PeerSummaryReader
from summariser.utils.reader import readSummaries
from pseudo_pyramid.similarity_scorer import parse_documents
from pseudo_pyramid.sbert_score_metrics import get_token_vecs, get_sbert_score


if __name__ == '__main__':
    year = '08'
    topn = 15
    ref_metric = 'top{}_10'.format(topn)
    sbert_type = 'f1'

    out_base = os.path.join(FEATURE_DIR,year)
    if not os.path.exists(out_base):
        os.makedirs(out_base)

    corpus_reader = CorpusReader(BASE_DIR)
    bert_model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')  # 'bert-large-nli-stsb-mean-tokens')

    sent_nums = []
    for topic,docs,models in tqdm(corpus_reader(year)):
        # encode docs
        if '.B' in topic: continue
        fpath = os.path.join(out_base, topic, 'sbert_top{}_sts_{}'.format(topn,sbert_type))
        if os.path.exists(fpath): continue
        print('\n=====topic {}====='.format(topic))
        sent_info_dic, sent_vecs, sents_weights = parse_documents(docs, bert_model, ref_metric)
        ref_dic = {k: sent_info_dic[k] for k in sent_info_dic if sents_weights[k] >= 0.1}
        print('extracted sent ratio', len(ref_dic)*1./len(sent_info_dic))
        # group sents by which article they are extracted from
        ref_sources = set(ref_dic[k]['doc'] for k in ref_dic)
        ref_sents = []
        for rs in ref_sources:
            ref_sents.append([ref_dic[k]['text'] for k in ref_dic if ref_dic[k]['doc']==rs])
        # get ref vectors and tokens
        ref_vecs = []
        ref_tokens = []
        for rsents in ref_sents:
            vv, tt = get_token_vecs(bert_model, rsents)
            ref_vecs.append(vv)
            ref_tokens.append(tt)
        # get sents in system summaries
        summ_vecs = []
        summ_tokens = []
        summs, ref_values_dic = readSummaries(year,topic,'rouge',sample_num=10000)
        for sids in tqdm(summs,desc='processing summaries'):
            ss = [sent_info_dic[ii]['text'] for ii in sids]
            vv, tt = get_token_vecs(bert_model, ss)
            summ_vecs.append(vv)
            summ_tokens.append(tt)
        # get sbert-score
        rewards = get_sbert_score(ref_vecs, ref_tokens, summ_vecs, summ_tokens, sbert_type)

        out_str = ''
        for ii, vv in enumerate(rewards):
            out_str += '{}\t{}\n'.format(summs[ii], vv)

        if not os.path.exists(os.path.join(out_base, topic)):
            os.makedirs(os.path.join(out_base, topic))

        ff = open(fpath, 'w')
        ff.write(out_str)
        ff.close()




