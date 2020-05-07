import sys
sys.path.append('../')

import os
from collections import OrderedDict
from nltk.tokenize import sent_tokenize

from resources import BASE_DIR

class CorpusReader:
    def __init__(self,topic_path,base_path=BASE_DIR,):
        self.base_path = base_path
        self.data_path = os.path.join(self.base_path,topic_path)

    def __call__(self):
        docs_dic = self.readDocs()
        return docs_dic

    def readSummaries(self):
        summary_path = os.path.join(self.data_path, 'summaries')
        summaries = []
        for tt in sorted(os.listdir(summary_path)):
            if tt[0] == '.':
                continue # skip hidden files
            text = open(os.path.join(summary_path,tt), 'r').read()
            summaries.append( text ) 
        return summaries

    def readReferences(self):
        ref_path = os.path.join(self.data_path, 'references')
        refs = []
        for tt in sorted(os.listdir(ref_path)):
            if tt[0] == '.':
                continue # skip hidden files
            text = open(os.path.join(ref_path,tt), 'r').read()
            refs.append( (os.path.join(ref_path,tt), sent_tokenize(text)) ) 
        return refs


    def readDocs(self):
        dpath = os.path.join(self.data_path,'input_docs')
        topic_docs = []
        for tt in sorted(os.listdir(dpath)):
            if tt[0] == '.':
                continue # skip hidden files
            entry = self.readOneDoc(os.path.join(dpath,tt))
            topic_docs.append((os.path.join(dpath,tt),entry))

        return topic_docs

    def readOneDoc(self,dpath):
        ff = open(dpath,'r')
        flag = False
        text = []
        for line in ff.readlines():
            if '<TEXT>' in line:
                flag = True
            elif '</TEXT>' in line:
                break
            elif flag and line.strip().lower() != '<p>' and line.strip().lower() != '</p>':
                text.append(line.strip())

        ff.close()

        return sent_tokenize(' '.join(text))


if __name__ == '__main__':
    reader = CorpusReader(BASE_DIR)
    docs = reader('data/topic_1')
    summs = reader.readSummaries()
    refs = reader.readReferences()

    print(docs)
    print(summs)
    print(refs)

