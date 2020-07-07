import os
from collections import OrderedDict

from resources import BASE_DIR

class PeerSummaryReader:
    def __init__(self,base_path):
        self.base_path = base_path

    def __call__(self,year):
        assert '08' == year or '09' == year
        data_path = os.path.join(self.base_path,'data','human_evaluations','UpdateSumm{}_eval'.format(year),'ROUGE','peers')
        summ_dic = self.readPeerSummary(data_path)

        return summ_dic

    def readPeerSummary(self,mpath):
        peer_dic = OrderedDict()

        for peer in sorted(os.listdir(mpath)):
            topic = self.uniTopicName(peer)
            if topic not in peer_dic:
                peer_dic[topic] = []
            sents = self.readOnePeer(os.path.join(mpath,peer))
            peer_dic[topic].append((os.path.join(mpath,peer),sents))

        return peer_dic

    def readOnePeer(self,mpath):
        ff = open(mpath,'r',encoding='latin-1')
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



if __name__ == '__main__':
    peerReader = PeerSummaryReader(BASE_DIR)
    summ = peerReader('08')

    for topic in summ:
        print('topic {}, summ {}'.format(topic,summ[topic]))