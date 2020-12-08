from resources import NER_DIR
import os
import ast

coreNLP_NER_tags = ["PERSON", "LOCATION", "ORGANIZATION", "MISC", "MONEY", "NUMBER", "ORDINAL", "PERCENT",
                    "DATE", "TIME", "DURATION", "SET",
                    "EMAIL", "URL", "CITY", "STATE_OR_PROVINCE", "COUNTRY", "NATIONALITY", "RELIGION", "TITLE",
                    "IDEOLOGY", "CRIMINAL_CHARGE", "CAUSE_OF_DEATH"]
coreNLP_group_names = ["PERSON", "LOCATION", "ORGANIZATION", "MISC", "MONEY", "NUMBER", "ORDINAL", "PERCENT",
                       "DATE", "TIME", "DURATION", "SET",
                       "EMAIL", "URL", "CITY", "STATE-OR-PROVINCE", "COUNTRY", "NATIONALITY", "RELIGION", "TITLE",
                       "IDEOLOGY", "CRIMINAL-CHARGE", "CAUSE-OF-DEATH"]
def read_ner(dataset,topic):
    file_path = os.path.join(NER_DIR,dataset,topic, 'docs.ner')
    docs = []
    for file in os.listdir(file_path):
        with open(os.path.join(file_path, file)) as f:
            doc = []
            lines = f.readlines()
            for line in lines:
                doc.append(ast.literal_eval(line))
            docs.append(doc)
    return docs

def parse_ner(dataset, topic, exclude=[]):
    docs = read_ner(dataset, topic)
    ner = {}
    for doc in docs:
        for line in doc:
            for tuple in line:
                if tuple[1] is not 'O' and tuple[1] not in exclude:
                    ner[tuple[0].lower()] = tuple[1]
    return ner

def parse_ner_distribution(dataset, topic):
    docs = read_ner(dataset, topic)
    ner = {}
    for doc in docs:
        for line in doc:
            for tuple in line:
                if tuple[1] is not 'O':
                    token = tuple[0].lower()
                    if token in ner:
                        ner[token] = ner[token]+1
                    else:
                        ner[token] = 1
    return ner

def parse_ner_chunk(dataset, topic, exclude=[]):
    docs = read_ner(dataset, topic)
    ner = {}
    for doc in docs:
        current_ner = []
        for line in doc:
            for t in line:
                if current_ner and t[1] != current_ner[0][1]:
                    ner[(tuple([s.lower() for s, _ in current_ner]))] = current_ner[0][1]
                    current_ner = []
                if (current_ner and t[1] == current_ner[0][1]) or (not current_ner and t[1] is not 'O'):
                    current_ner.append(t)
    return ner

def parse_ner_chunk_distribution(dataset, topic, exclude=[]):
    docs = read_ner(dataset, topic)
    ner = {}
    for doc in docs:
        current_ner = []
        for line in doc:
            for t in line:
                if current_ner and t[1] != current_ner[0][1]:
                    addition = tuple([s.lower() for s, _ in current_ner])
                    if addition in ner:
                        ner[addition] = ner[addition]+1
                    else:
                        ner[addition] = 1
                    current_ner = []
                if (current_ner and t[1] == current_ner[0][1]) or (not current_ner and t[1] is not 'O'):
                    current_ner.append(t)
    return ner

def parse_ner_chunk_df(dataset, topic):
    docs = read_ner(dataset, topic)
    ner = {}
    for doc in docs:
        current_ner = []
        found_in_doc = set()
        for line in doc:
            for t in line:
                if current_ner and t[1] != current_ner[0][1]:
                    addition = tuple([s.lower() for s, _ in current_ner])
                    if addition not in found_in_doc:
                        found_in_doc.add(addition)
                        if addition in ner:
                            ner[addition] = ner[addition]+1
                        else:
                            ner[addition] = 1
                    current_ner = []
                if (current_ner and t[1] == current_ner[0][1]) or (not current_ner and t[1] is not 'O'):
                    current_ner.append(t)
    ner = {entity : ner[entity]/float(len(docs)) for entity in ner.keys()}
    return ner

def parse_ner_chunk_grouped(dataset, topic):
    ner = parse_ner_chunk(dataset, topic)
    ner_df = parse_ner_chunk_df(dataset, topic)

    grouped_df = {tag:{} for tag in coreNLP_NER_tags}

    for chunk in ner.keys():
        grouped_df[ner[chunk]][chunk] = ner_df[chunk]
    return grouped_df, coreNLP_NER_tags
