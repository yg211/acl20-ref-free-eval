import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #os.path.abspath('.')
ROUGE_DIR = os.path.join(BASE_DIR,'summariser','rouge','ROUGE-RELEASE-1.5.5/') #do not delete the '/' in the end
SUMMARY_DB_DIR = os.path.join(BASE_DIR,'data','sampled_summaries')
FEATURE_DIR = os.path.join(BASE_DIR,'data','summary_feature')
NER_DIR = os.path.join(BASE_DIR,'data','ner')

EMBEDDING_PATH = '/home/cim/staff/uhac002/Library/Embeddings'
GLOVE_PATH = os.path.join(EMBEDDING_PATH,'GloVe','glove.840B.300d.txt')
DBOW_PATH = os.path.join(EMBEDDING_PATH,'apnews_doc2vec','doc2vec.bin')
INFERSENT_PATH = os.path.join(EMBEDDING_PATH,'InferSent','infersent2.pkl')
W2V_PATH = os.path.join(EMBEDDING_PATH,'fastText','crawl-300d-2M.vec')
DOC_SEQUENCE_PATH = os.path.join(BASE_DIR,'summariser','utils','DocsSequence.txt')

SUMMARY_LENGTH = 100
LANGUAGE = 'english'

EMBED_SCORE_PATH = os.path.join(BASE_DIR,'embed_score')
ROBERTA_EMBEDDING_PATH = os.path.join(EMBED_SCORE_PATH,'roberta','roberta_embeddings')
BERT_EMBEDDING_PATH = os.path.join(EMBED_SCORE_PATH,'bert','bert_embeddings')
STS_BERT_EMBEDDING_PATH = os.path.join(EMBED_SCORE_PATH,'sts_bert','sts_bert_embeddings')
NLI_BERT_EMBEDDING_PATH = os.path.join(EMBED_SCORE_PATH,'nli_bert','nli_bert_embeddings')

