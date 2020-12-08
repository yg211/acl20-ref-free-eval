import csv
from collections import OrderedDict
import os, shutil
import codecs

def create_dir(dirpath):
    p = os.path.normpath(os.path.expanduser(dirpath))
    if not os.path.isdir(p):
        os.makedirs(p)
    return p

def clean_create_dir(dirpath):
    if os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)

def write_rl_details(summary, model_name_list, R1_list, R2_list, R4_list, filename):
    '''
    YG: to write the results of the RL-based summarizer
    :param summary: a list of sentences (strings)
    :param R1:
    :param R2:
    :param R4:
    :param filename:
    :return: null
    '''
    dirpath = os.path.dirname(filename)
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
    if len(summary) == 0:
        return

    for i in range(len(model_name_list)):
        model_name = model_name_list[i]
        text = '\nmodel name: {} \n'.format(model_name)
        text += '\tR1 : {} ;'.format(R1_list[i])
        text += 'R2 : {} ;'.format(R2_list[i])
        text += 'R4 : {} ;'.format(R4_list[i])

    text += '\n'
    for sent in summary:
        text +=  sent + '\n'

    write_to_file(text, filename)


def write_details_file(details_list,filename):
    '''
    [[0, R1, R2, text], [1, R1, R2, text]], [[0, R1, R2, text], [1, R1, R2, text]]
    '''
    dirpath = os.path.dirname(filename)
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
    if len(details_list) == 0:
        return

    max_len = max([len(feedback) for feedback in details_list])
    text = ''
    for i in range(max_len):
        text += '%s,' % (i)
        for feedback in details_list:
            if i >= len(feedback):
                text += ",,,,,,"
            else:
                _, R1, R2, SU4, accept, reject, summ_text = feedback[i]
                text += u"%s,%s,%s,%s,%s,\"%s\"," % (str(R1), str(R2), str(SU4), str(accept), str(reject), unicode(summ_text))
        text = text[:-1] + '\n'
    write_to_file(text, filename)
        
def write_config(filename, config):
    string = "<ROUGE-EVAL version=\"1.55\">\n" + config + "</ROUGE-EVAL>\n"
    write_to_file(string, filename)

def write_to_csv(result_dict, filename):
    ordered_dict = OrderedDict(sorted(result_dict.items(), key=lambda t: t[0]))
    with open(filename, 'wb') as csvfile:
        fieldnames = ["Experiment","ROUGE-1 R","ROUGE-1 P","ROUGE-1 F","ROUGE-2 R","ROUGE-2 P",\
                      "ROUGE-2 F", "ROUGE-3 R", "ROUGE-3 P", "ROUGE-3 F", "ROUGE-4 R",\
                       "ROUGE-4 P", "ROUGE-4 F","ROUGE-SU4 F", "ROUGE-SU4 P", "ROUGE-4 F"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for key in ordered_dict:
            writer.writerow(ordered_dict[key])

def write_to_file(text, filename, add=False):
    if not add:
        fp = open(filename,'w')
    else:
        fp = open(filename,'a')
    fp.write(text)
    fp.close()


def append_to_file(text, filename):
    if os.path.isfile(filename):
        fp = open(filename,'a')
    else:
        fp = open(filename,'w+')
    fp.write(text)
    fp.close()