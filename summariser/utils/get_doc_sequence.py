import os
from resources import PROCESSED_PATH

datasets = os.listdir(PROCESSED_PATH)
output = ''

for dataset in datasets:
    dataset_path = os.path.join(PROCESSED_PATH,dataset)
    topics = os.listdir(dataset_path)
    for topic in topics:
        docs = os.listdir(os.path.join(dataset_path,topic,'docs'))
        output += '{};{} : '.format(dataset,topic)
        for dd in docs:
            output += '{};'.format(dd)
        output = output[:-1]+'\n'

out_file = 'DocsSequence.txt'
ff = open(out_file,'w')
ff.write(output)
ff.close()

