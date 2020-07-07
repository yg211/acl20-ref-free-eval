import os
from collections import OrderedDict
from nltk.tokenize import sent_tokenize

from resources import BASE_DIR

class CorpusReader:
    def __init__(self,base_path):
        self.base_path = base_path

    def __call__(self,year):
        assert '08' == year or '09' == year
        data_path = os.path.join(self.base_path,'data','input_docs','UpdateSumm{}_test_docs_files'.format(year))
        model_path = os.path.join(self.base_path,'data','human_evaluations','UpdateSumm{}_eval'.format(year),'ROUGE','models')
        corpus = []

        docs_dic = self.readDocs(data_path)
        models_dic = self.readModels(model_path)

        for topic in docs_dic:
            entry = []
            entry.append(topic)
            entry.append(docs_dic[topic])
            entry.append(models_dic[topic])
            corpus.append(entry)

        return corpus

    def readModels(self,mpath):
        model_dic = OrderedDict()

        for model in sorted(os.listdir(mpath)):
            topic = self.uniTopicName(model)
            if topic not in model_dic:
                model_dic[topic] = []
            sents = self.readOneModel(os.path.join(mpath,model))
            model_dic[topic].append((os.path.join(mpath,model),sents))

        return model_dic

    def readOneModel(self,mpath):
        ff = open(mpath,'r')
        sents = []
        for line in ff.readlines():
            if line.strip() != '':
                sents.append(line.strip())
        ff.close()
        return sents

    def uniTopicName(self,name):
        doc_name = name.split('-')[0][:5]
        block_name = name.split('-')[1][0]
        return '{}.{}'.format(doc_name,block_name)

    def readDocs(self,dpath):
        data_dic = OrderedDict()

        for tt in sorted(os.listdir(dpath)):
            if tt[0] == '.':
                continue
            for topic in sorted(os.listdir(os.path.join(dpath,tt))):
                topic_docs = []
                doc_names = sorted(os.listdir(os.path.join(dpath,tt,topic)))
                for doc in doc_names:
                    entry = self.readOneDoc(os.path.join(dpath,tt,topic,doc))
                    topic_docs.append((os.path.join(dpath,tt,topic,doc),entry))
                data_dic[self.uniTopicName(topic)] = topic_docs

        return data_dic

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
    data = reader('08')

    for topic,docs,models in data:
        print('\n---topic {}, docs {}, models {}'.format(topic,docs[0],models[0]))
