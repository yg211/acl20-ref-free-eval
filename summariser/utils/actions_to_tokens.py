from summariser.utils.corpus_reader import CorpusReader
from resources import PROCESSED_PATH, BASE_DIR#, W2V_PATH
from summariser.utils.reader import readSummaries
from summariser.utils.data_helpers import sent2tokens
import pickle
import os
#from gensim.models import KeyedVectors
import numpy as np

TOKEN2IDX_FILE = os.path.join(BASE_DIR,'data', 'duc_token2idx.pkl')
FASTTEXT_E_FILE = os.path.join(BASE_DIR, 'data', 'duc_fastText_embedding.pkl')
W2V_E_FILE = os.path.join(BASE_DIR, 'data', 'duc_w2v_embedding.pkl')
def build_duc_vocabulary():
    datasets = ['DUC2001', 'DUC2002', 'DUC2004']
    sample_num = 9999
    cv_fold_num = 10
    validation_size = 0.1

    ### read documents and ref. summaries
    reader = CorpusReader(PROCESSED_PATH)
    vocabulary = set()
    for dataset in datasets:
        data = reader.get_data(dataset)
        ### read data
        for topic,docs,models in data:
            print('read DATA {}, TOPIC {}'.format(dataset,topic))
            summs, ref_values_dic = readSummaries(dataset,topic,'rouge',sample_num)
            sentences = [sent2tokens(sentence, 'english') for _, doc in docs for sentence in doc]
            vocabulary.update([token for sentence in sentences for token in sentence])
    return vocabulary

def build_duc_token2idx(write):
    vocab = build_duc_vocabulary()
    token2idx = {token : i+1 for i, token in enumerate(vocab)}
    if write:
        pickle.dump(token2idx, open(TOKEN2IDX_FILE, "wb"), protocol=2)
    return token2idx

def read_duc_token2idx():
    t2i = pickle.load(open(TOKEN2IDX_FILE, 'rb'))
    return t2i

def actions_to_idx(summaries, groups, sentences_of_topics, token2idx):
    return [[token2idx[token] for action in summary for token in sentences_of_topics[groups[i]][action]] for i, summary in enumerate(summaries)]

def build_embedding():
    fastText = KeyedVectors.load_word2vec_format(W2V_PATH)
    token2idx = pickle.load(open(TOKEN2IDX_FILE, "rb"))

    fastText_embedding = np.zeros((len(token2idx)+1, 300))
    hits = 0
    for token in token2idx.keys():
        try:
            fastText_embedding[token2idx[token]] = fastText[token]
            hits += 1
        except KeyError:
            fastText_embedding[token2idx[token]] = np.random.normal(size=300)
    pickle.dump(fastText_embedding, open(FASTTEXT_E_FILE, "wb"), protocol=2)
    print(hits/len(token2idx))

def build_w2v_embedding():
    wv2 = KeyedVectors.load_word2vec_format(W2V_PATH, binary=True, limit=1000000)
    token2idx = pickle.load(open(TOKEN2IDX_FILE, "rb"))

    w2v_embedding = np.zeros((len(token2idx)+1, 300))
    hits = 0
    for token in token2idx.keys():
        try:
            w2v_embedding[token2idx[token]] = wv2[token]
            hits += 1
        except KeyError:
            w2v_embedding[token2idx[token]] = np.random.normal(size=300)
    pickle.dump(w2v_embedding, open(W2V_E_FILE, "wb"), protocol=2)
    print(hits/len(token2idx))

def read_duc_fastText_embedding():
    return pickle.load(open(FASTTEXT_E_FILE, "rb"))

def read_duc_w2v_embedding():
    return pickle.load(open(W2V_E_FILE, "rb"))

if __name__ == "__main__":
    build_embedding()
