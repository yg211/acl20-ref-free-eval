import csv, codecs
import os
from resources import *
from summariser.utils.misc import *
   
def read_csv(filename):
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar= '"')
        rows = [row for row in reader]
        return rows

def read_file(filename):
    with codecs.open(filename, 'r', 'utf-8', errors='ignore') as fp:
        return fp.read()

def readSampleSummaries(dataset,topic,sample_num=9999):
    summaries, ref_values = readSummaries(dataset,topic,'rouge',sample_num)
    summaries, heu_values = readSummaries(dataset,topic,'heuristic',sample_num)

    return summaries, ref_values, heu_values

def readSummaries(dataset,topic,reward_type='rouge',sample_num=9999,raw=False):
    path = os.path.join(SUMMARY_DB_DIR,dataset,topic,reward_type)
    summary_list = []
    value_list = []
    model_names = []

    with open(path,'r') as ff:
        if reward_type == 'heuristic' or 'js' in reward_type:
            cnt = 0
            while cnt < sample_num:
                line = ff.readline()
                if line == '':
                    break
                if 'actions' not in line:
                    continue
                else:
                    cnt += 1
                    nums_str = line.split(':')[1].split(',')
                    acts = []
                    for nn in nums_str:
                        acts.append(int(nn))
                    summary_list.append(acts)
                    value = float(ff.readline().split(':')[1])
                    value_list.append(value)
        elif reward_type == 'rouge':
            flag = False
            value = []
            idx = -1
            cnt = 0
            while cnt < sample_num:
                line = ff.readline()
                if line == '':
                    break
                if 'model' not in line and not flag:
                    continue
                elif 'model' in line:
                    idx = int(line.split(':')[0].split(' ')[1])
                    name = line.split(':')[1].strip()
                    if name not in model_names:
                        model_names.append(name)
                    flag = True
                elif 'model' not in line and flag:
                    if 'action' in line and idx == 0:
                        nums_str = line.split(':')[1].split(',')
                        acts = []
                        for nn in nums_str:
                            acts.append(int(nn))
                        summary_list.append(acts)
                    elif 'R1' in line:
                        scores = line.split(';')
                        R1 = float(scores[0].split(':')[1])
                        R2 = float(scores[1].split(':')[1])
                        # R3 = float(scores[2].split(':')[1])
                        # R4 = float(scores[3].split(':')[1])
                        # RL = float(scores[4].split(':')[1])
                        RSU = float(scores[5].split(':')[1])
                        vv= (R1/0.48 + R2/0.212 + RSU/0.195)
                        value.append(vv)
                    elif 'action' not in line and 'R1' not in line:
                        flag = False
                        value_list.append(value)
                        value = []
                        cnt += 1

    # normalise
    norm_value_dic = {}
    if reward_type == 'heuristic':
        if raw:
            norm_value_dic = value_list
        else:
            norm_value_dic = normaliseList(value_list)
    elif 'js' in reward_type:
        if raw:
            norm_value_dic = value_list
        else:
            max_js = max(value_list)
            rewards = []
            for js in value_list:
                rewards.append(max_js-js)
            norm_value_dic = normaliseList(rewards)
    else:
        assert reward_type == 'rouge'
        for ii in range(len(value_list[0])):
            temp_list = [x[ii] for x in value_list]
            if raw:
                norm_value_dic[model_names[ii]] = temp_list
            else:
                norm_value_dic[model_names[ii]] = normaliseList(temp_list)

    return summary_list, norm_value_dic