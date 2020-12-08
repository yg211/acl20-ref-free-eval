import pandas as pd
from resources import BASE_DIR
import os
import numpy as np

from summariser.utils.evaluator import plotAgreement

class HumanMetricsData:
    def __init__(self):
        fpath = os.path.join(BASE_DIR,'data','human_evaluation','lqual.jsonl')
        self.df = pd.read_json(fpath,lines=True)

    def getSummaryIndices(self):
        id_list = np.array(self.df['id'])
        return list(set(id_list))

    def getEntriesbyIndex(self,idx):
        entries = self.df[self.df['id']==idx]
        return entries

    def getColumns(self):
        return np.array(self.df.columns)

if __name__ == '__main__':

    data = HumanMetricsData()
    summaries = data.getSummaryIndices()
    print('col: {}'.format(data.getColumns()))

    gold_list = []
    rouge2_list = []
    for sid in summaries:
        entries = data.getEntriesbyIndex(sid)
        for system in entries['prompts']:
            gold_list.append(system['overall']['gold'])
            rouge2_list.append(system['overall']['ROUGE-2'])

    plotAgreement(gold_list,rouge2_list,5,True)


