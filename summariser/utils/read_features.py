import numpy as np
import os,re
from resources import FEATURE_DIR,SUMMARY_DB_DIR
from summariser.utils.read_ner import coreNLP_group_names

def parseFeatures(path,test):
    ff = open(path,'r')
    values = []
    summaries = []
    for idx, line in enumerate(ff.readlines()):
        if line.strip() == '':
            continue
        eles = line.split('\t')
        if test is not None:
            if idx >= len(test) or not '{}'.format(test[idx])==eles[0]:
                break
        else:
            acts = [int(ee.replace(']','').replace('[','')) for ee in eles[0].split(',')]
            summaries.append(acts)
        if len(eles) == 2:
            values.append(float(eles[1]))
        elif len(eles) > 2:
            temp = []
            for vv in eles[1:]:
                if 'nan' in vv:
                    temp.append(0.0)
                else:
                    temp.append(float(re.findall('\d+\.\d+',vv)[0]))
            values.append(temp)
    ff.close()

    if test is not None:
        return np.array(values)
    else:
        return summaries, values

def parseHeuristic(path,test):
    ff = open(path,'r')
    values = []
    for line in ff.readlines():
        if line.strip() == '':
            continue
        if 'actions' in line:
            if test is not None and ','.join(format(x) for x in test[len(values)]) not in line:
                break
        elif 'utility' in line:
            values.append(float(line.split(':')[1]))
            if len(values) >= len(test):
                break
    ff.close()
    return np.array(values)

def readFeatures(feature_types,dataset,summaries,groups,wanted_groups):
    features = None
    for type in feature_types:
        single_feature = []
        for topic in wanted_groups:
            #if topic != 'D0848.A': continue
            #print(type,topic)
            train = np.array([topic in gg for gg in groups])
            if type != 'heuristic':
                if 'infersent' in type and 'all_docs' not in type and 'heuristic' not in type:
                    file_path = os.path.join(FEATURE_DIR,dataset,topic,'infersent_min_mean_max_std')
                    rff = parseFeatures(file_path,summaries[train])
                    nn = ['min','mean','max','std']
                    cols = [n in type for n in nn]
                    ff = rff[:,cols]
                elif type == 'ner_recall' or type == 'ner_density' or type == 'ner_token_recall' or type == 'ner_token_density':
                    if 'token' in type:
                        file_path = os.path.join(FEATURE_DIR,dataset,topic,'ner_token_recall_density')
                    else:
                        file_path = os.path.join(FEATURE_DIR,dataset,topic,'ner_recall_density')
                    rff = parseFeatures(file_path,summaries[train])
                    if 'recall' in type:
                        ff = rff[:,0]
                    else:
                        ff = rff[:,1]
                elif type == 'tf' or type == 'cf':
                    file_path = os.path.join(FEATURE_DIR,dataset,topic,'tf_cf')
                    rff = parseFeatures(file_path,summaries[train])
                    if type == 'tf':
                        ff = rff[:,0]
                    else:
                        ff = rff[:,1]
                elif type == 'redundancy_1' or type == 'redundancy_2':
                    file_path = os.path.join(FEATURE_DIR,dataset,topic,'redundancy_1_2')
                    rff = parseFeatures(file_path,summaries[train])
                    if '1' in type:
                        ff = rff[:,0]
                    else:
                        ff = rff[:,1]
                elif type == 'coherence_mean' or type == 'coherence_std':
                    file_path = os.path.join(FEATURE_DIR,dataset,topic,'coherence_mean_std')
                    rff = parseFeatures(file_path,summaries[train])
                    if 'mean' in type:
                        ff = rff[:,0]
                    else:
                        ff = rff[:,1]
                elif 'ner_grouped_' in type:
                    tags = type.split('_')[2:]
                    cols = [coreNLP_group_names.index(group) for group in tags]
                    file_path = os.path.join(FEATURE_DIR,dataset,topic,'ner_grouped')
                    rff = parseFeatures(file_path,summaries[train])
                    ff = rff[:,cols]
                else:
                    file_path = os.path.join(FEATURE_DIR,dataset,topic,type)
                    ff = parseFeatures(file_path,summaries[train])
            else:
                file_path = os.path.join(SUMMARY_DB_DIR,dataset,topic,type)
                ff = parseHeuristic(file_path,summaries[train])
            single_feature.extend(ff)
        if features is None:
            features = np.copy(single_feature)
            features = np.array([features]).reshape(len(single_feature),-1)
        else:
            features = np.c_[features,single_feature]

    return features

