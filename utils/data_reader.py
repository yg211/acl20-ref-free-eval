import sys
sys.path.append('../')

import os
from collections import OrderedDict
from nltk.tokenize import sent_tokenize

from resources import BASE_DIR

class CorpusReader:
    def __init__(self,base_path):
        self.base_path = base_path

    def __call__(self, topic_path):
        data_path = os.path.join(self.base_path,topic_path)
        docs_dic = self.readDocs(data_path)
        return docs_dic

    def readDocs(self,dpath):
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
    docs_dict = reader('input_docs/sample_topic')

    print(docs_dict)

