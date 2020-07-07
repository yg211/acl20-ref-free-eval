import os
from collections import OrderedDict
import xmltodict

class TacData:
    def __init__(self,base_dir,year):
        assert str(year) in ['08','09']
        self._year = year
        self._basedir = os.path.join(base_dir,'data','human_evaluations','UpdateSumm{}_eval'.format(year))

    def getHumanScores(self,level,metric):
        if level == 'system' or level == 'macro':
            system_manual_scores = self._getAllSystemManualScores()
        elif level == 'summary' or level == 'micro':
            system_manual_scores = self._getAllSummaryManualScores()
        ss = OrderedDict()
        assert metric in ['scu','pyramid','lingustic','responsiveness']
        for summ in system_manual_scores:
            ss[summ] = system_manual_scores[summ][metric]

        return ss

    def getRougeScores(self,metric):
        rouge_scores = self._getAllRougeScores()
        assert metric in ['rouge2','rougeSU4']
        rr = OrderedDict()

        for summ in rouge_scores:
            rr[summ] = rouge_scores[summ][metric]

        return rr

    def getAllSCUs(self):
        path = os.path.join(self._basedir,'manual','pyramids')
        index_dic = OrderedDict()
        for fname in os.listdir(path):
            with open(os.path.join(path,fname),'r') as fd:
                index_dic[fname] = xmltodict.parse(fd.read())

        return index_dic

    def _getAllRougeScores(self):
        scores = OrderedDict()
        rouge_base = os.path.join(self._basedir,'ROUGE')
        fnames = os.listdir(rouge_base)

        for name in fnames:
            if 'jk' not in name  or 'avg' not in name:
                continue
            self._readRougeScores(scores,os.path.join(rouge_base,name))

        return scores

    def _readRougeScores(self,dic,fpath):
        metric = fpath.split('/')[-1].split('.')[0].split('_')[0]
        if self._year == '09':
            block = fpath.split('/')[-1].split('.')[0].split('_')[1]

        ff = open(fpath,'r')
        for line in ff.readlines():
            if line.strip() == '':
                continue
            eles = line.split(' ')
            if self._year == '09':
                key = 'block{}_sum{}'.format(block,eles[0])
            else:
                key = 'sum{}'.format(eles[0])
            if key not in dic:
                dic[key] = OrderedDict()
            dic[key][metric] = float(eles[1])

        ff.close()


    def _getAllSystemManualScores(self):
        scores = OrderedDict()
        manual_base = os.path.join(self._basedir,'manual')
        fnames = os.listdir(manual_base)
        for name in fnames:
            if '.avg' not in name:
                continue
            if 'model' in name:
                if name.split('.')[-2] in ['A','B']:
                    self._readAvgManualScores(scores,os.path.join(manual_base,name),name.split('.')[-2])
                else:
                    self._readAvgManualScores(scores,os.path.join(manual_base,name))
            elif 'peer' in name:
                if name.split('.')[-2] in ['A','B']:
                    self._readAvgPeerScores(scores,os.path.join(manual_base,name),name.split('.')[-2])
                else:
                    self._readAvgPeerScores(scores,os.path.join(manual_base,name))
        return scores

    def _getAllSummaryManualScores(self):
        scores = OrderedDict()
        manual_base = os.path.join(self._basedir, 'manual')
        fnames = os.listdir(manual_base)
        for name in fnames:
            if '.avg' in name: continue
            if name[0] == '.': continue # exclude hidden files
            if '.' not in name: continue # eclude directories
            if 'model' in name:
                if 'A' in name or 'B' in name:
                    self._readManualScores(scores, os.path.join(manual_base, name), name.split('.')[-2])
                else:
                    self._readManualScores(scores, os.path.join(manual_base, name))
            elif 'peer' in name:
                if 'A' in name or 'B' in name:
                    self._readPeerScores(scores, os.path.join(manual_base, name), name.split('.')[-2])
                else:
                    self._readPeerScores(scores, os.path.join(manual_base, name))
        return scores



    def _readAvgPeerScores(self,dic,fpath,block=None):
        metrics = ['pyramid','scu','rep','macroavg','linguistic','responsiveness']
        ff = open(fpath,'r')
        for line in ff.readlines():
            if line.strip() == '':
                continue
            eles = line.split('\t')
            if block is None:
                key = 'sum{}'.format(eles[0])
            else:
                key = 'block{}_sum{}'.format(block,eles[0])
            new_entry = OrderedDict()
            for i in range(1,len(eles)):
                new_entry[metrics[i-1]] = float(eles[i])
            dic[key] = new_entry

        ff.close()

    def _readPeerScores(self,dic,fpath,block=None):
        metrics = ['pyramid','scu','rep','macroavg','linguistic','responsiveness']
        ff = open(fpath,'r')
        for line in ff.readlines():
            if line.strip() == '':
                continue
            if block is None:
                eles = line.split(' ')
                idx_list = [2,3,4,6,7,8]
            else:
                eles = line.split('\t')
                idx_list = [2,3,4,7,8,9]
            topic = eles[0]
            key = 'topic{}_'.format(topic)
            key += 'sum{}'.format(eles[1])
            #if block is None:
            #    key += 'sum{}'.format(eles[1])
            #else:
            #    key += 'block{}_sum{}'.format(block,eles[1])
            new_entry = OrderedDict()
            for i,j in enumerate(idx_list):
                new_entry[metrics[i]] = float(eles[j])
            dic[key] = new_entry

        ff.close()


    def _readAvgManualScores(self,dic,fpath,idx,block=None):
        metrics = ['scu','pyramid','lingustic','responsiveness']
        ff = open(fpath,'r')
        for line in ff.readlines():
            if line.strip() == '':
                continue
            eles = line.split('\t')
            if block is None:
                key = 'sum{}'.format(eles[0])
            else:
                key = 'block{}_sum{}'.format(block,eles[0])
            new_entry = OrderedDict()
            for i in range(1, len(eles)):
                new_entry[metrics[i]] = float(eles[j])
            dic[key] = new_entry

        ff.close()

    def _readManualScores(self,dic,fpath,block=None):
        metrics = ['scu','pyramid','lingustic','responsiveness']
        ff = open(fpath,'r')
        for line in ff.readlines():
            if line.strip() == '':
                continue
            if block is None:
                eles = line.split(' ')
                idx_list = [2,4,5,6]
            else:
                eles = line.split('\t')
                idx_list = [2,5,6,7]
            key = 'topic{}_'.format(eles[0])
            if block is None:
                key += 'sum{}'.format(eles[0])
            else:
                key += 'block{}_sum{}'.format(block,eles[0])
            new_entry = OrderedDict()
            for i,j in enumerate(idx_list):
                new_entry[metrics[i]] = float(eles[j])
            dic[key] = new_entry

        ff.close()

