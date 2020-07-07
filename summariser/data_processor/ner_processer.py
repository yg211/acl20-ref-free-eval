import os

from summariser.utils.writer import write_to_file
from summariser.data_processor.corpus_reader import CorpusReader
from resources import NER_DIR, LANGUAGE, CORENLP_PATH, BASE_DIR
from summariser.utils.data_helpers import sent2stokens, sent2stokens_wostop
from nltk.stem.snowball import SnowballStemmer
from stanfordcorenlp import StanfordCoreNLP

def writeNER(dataset):
    summary_len = 100
    reader = CorpusReader(BASE_DIR)
    data = reader(dataset)
    nlp = StanfordCoreNLP(CORENLP_PATH)
    base_path = os.path.join(NER_DIR,dataset)

    topic_cnt = 0

    for topic, docs, models in data:
        if '.B' in topic:
            continue

        topic_cnt += 1
        if not os.path.exists(os.path.join(base_path,topic)):
            os.mkdir(os.path.join(base_path,topic))
        topic_path = os.path.join(base_path,topic,'docs.ner')
        if not os.path.exists(topic_path):
            os.mkdir(topic_path)
        for dd in docs:
            dname = dd[0].split('/')[-1].strip()
            print('{} topic {}, doc {}'.format(topic_cnt,topic,dname))
            output = ''
            for sen in dd[1]:
                print(sen)
                if sen is None:
                    continue
                ner = nlp.ner(sen)
                output += repr(ner)+'\n'
            write_to_file(output,os.path.join(topic_path,dname))

    nlp.close()


if __name__ == '__main__':
    ds = ['09']
    for dd in ds:
        print('\n==={}==='.format(dd))
        writeNER(dd)