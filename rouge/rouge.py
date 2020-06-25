import sys
sys.path.append('../')

from os import path,makedirs,getcwd
import time
import os
from utils.writer import write_to_file
from subprocess import check_output
import re
import shutil
import random


class Rouge(object):
    def __init__(self, rouge_dir, base_dir, rouge_l=True, verbose=False):
        self.ROUGE_DIR = rouge_dir
        self.reference_summary_temp_filename = "reference_summary.txt"
        config_file = "config.xml"
        self.temp_dir = os.path.join(base_dir,'rouge_temp_files',repr(random.random())[3:8]+'-'+repr(time.time()))
        self.temp_config_file = path.join(self.temp_dir, config_file)
        self.rouge_l = rouge_l
        self.verbose = verbose

        if not path.exists(self.temp_dir):
            makedirs(self.temp_dir)
        # print("created Rouge instance with tmp: '%s'" % self.temp_dir)
        # print("created Rouge instance with summary_file: '%s'" %  self.reference_summary_temp_filename)
        # print("created Rouge instance with config_file: '%s'" % self.temp_config_file)
        # print("created Rouge instance with ROUGE_DIR: ", self.ROUGE_DIR)

    def create_config(self, peers, models, models_dir):
        config_file = "<EVAL ID=\"1\">\n"
        config_file += "<PEER-ROOT>\n"
        config_file += self.temp_dir + "\n"
        config_file += "</PEER-ROOT>\n"
        config_file += "<MODEL-ROOT>\n"
        config_file += models_dir + "\n"
        config_file += "</MODEL-ROOT>\n"

        config_file += "<INPUT-FORMAT TYPE=\"SPL\">\n</INPUT-FORMAT>\n"
        config_file += "<PEERS>\n"
        for i, peer in enumerate(peers):
            config_file += "<P ID=\"" + str(i + 1) + "\">" + peer + "</P>\n"
        config_file += "</PEERS>\n"

        config_file += "<MODELS>\n"
        for model, _ in models:
            model_name = path.basename(model)
            config_file += "<M ID=\"" + model_name[-1] + "\">" + model_name + "</M>\n"
        config_file += "</MODELS>\n"
        config_file += "</EVAL>\n"

        return config_file

    def extract_results(self, result):
        lines = result.split("\n")
        #print('rouge result lines:\n')
        #for line in lines:
            #print(line)
        result_dict = {}
        prev_exp = ""
        for line in lines:
            x = re.search("([\w\d]+) (ROUGE-[\w\d][\w]?[*]?) Average_(\w): (\d\.\d*) .+", line)
            if x:
                exp_no, rouge_name, stype, score = x.group(1), x.group(2), x.group(3), x.group(4)
                index = exp_no
                rouge_type = rouge_name + " " + stype
                if exp_no != prev_exp:
                    if index not in result_dict:
                        result_dict[index] = {}
                    result_dict[index]["Experiment"] = exp_no
                    prev_exp = exp_no
                result_dict[index][rouge_type] = score
        return result_dict

    def execute_rouge(self):
        cmd = "perl " + self.ROUGE_DIR + "ROUGE-1.5.5.pl -e " + self.ROUGE_DIR + "data " + self.ROUGE_ARGS + ' -a ' + self.temp_config_file
        #print("execute_rouge command is" , cmd)
        
        return check_output(cmd, shell=True)

    def clean(self):
        shutil.rmtree(self.temp_dir)

    def get_scores(self, summary, models):
        write_to_file(summary, path.join(self.temp_dir, self.reference_summary_temp_filename),True)

        models_dir = path.dirname(models[0][0])
        config = self.create_config([self.reference_summary_temp_filename], models, models_dir)

        write_to_file(config, self.temp_config_file, True)

        result = self.execute_rouge()

        result_dict = self.extract_results(result.decode('utf-8'))

        R1score = float(result_dict["1"]['ROUGE-1 R'])
        R2score = float(result_dict["1"]['ROUGE-2 R'])
        if self.verbose:
            R3score = float(result_dict["1"]['ROUGE-3 R'])
            R4score = float(result_dict["1"]['ROUGE-4 R'])
        if self.rouge_l:
            RLscore = float(result_dict["1"]['ROUGE-L R'])
        RSU4score = float(result_dict["1"]['ROUGE-SU* R'])
        if self.verbose and self.rouge_l:
            return R1score, R2score, R3score, R4score, RLscore, RSU4score
        elif self.verbose and not self.rouge_l:
            return R1score, R2score, R3score, R4score, RSU4score
        elif not self.verbose and self.rouge_l:
            return R1score, R2score, RLscore, RSU4score
        else:
            return R1score, R2score, RSU4score

    def __call__(self, summary, models, summary_len):
        if self.rouge_l == False:
            self.ROUGE_ARGS = '-n 4 -m -x -c 95 -r 1000 -f A -p 0.5 -t 0 -a -2 -4 -u -l %s' % (summary_len)
        else:
            self.ROUGE_ARGS = '-n 4 -m -c 95 -r 1000 -f A -p 0.5 -t 0 -a -2 -4 -u -l %s' % (summary_len)

        #self.ROUGE_ARGS = '-m -s -p 0 %s' % (summary_len)

        return self.get_scores(summary, models)
