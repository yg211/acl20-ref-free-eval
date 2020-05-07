from __future__ import print_function, unicode_literals
import os.path as path
import sys

sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

import os
from nltk.tokenize import word_tokenize
from nltk import ngrams, RegexpParser
from nltk import pos_tag
from nltk import ne_chunk_sents
from nltk.tree import Tree
from collections import OrderedDict
import string, re

#from utils.loadEmbeddings import LoadEmbeddings

PUNCT = tuple(string.punctuation)

def remove_spaces_lines(text):
    '''
    Normalize text
    Remove & Replace unnessary characters
    Parameter argument:
    text: a string (e.g. '.... 
                        
                        New York N.Y is a city...')
    
    Return:
    text: a string (New York N.Y is a city.)
    '''
    text = re.sub('[\n\s\t_]+', ' ', str(text))
    return text

def text_normalization(text):
    '''
    Normalize text
    Remove & Replace unnessary characters
    Parameter argument:
    text: a string (e.g. '.... *** New York N.Y is a city...')
    
    Return:
    text: a string (New York N.Y is a city.)
    '''
    text = re.sub(u'\u201e|\u201c',u'', text)
    text = re.sub(u"\u2022",u'. ', text)  
    text = re.sub(u"([.?!]);",u"\\1", text)
    text = re.sub(u'``', u'``', text)
    text = re.sub(u"\.\.+",u" ", text)
    text = re.sub(u"\s+\.",u".", text)
    text = re.sub(u"\?\.",u"?", text)
    text = re.sub(u'[\n\s\t_]+',u' ', text)
    text = re.sub(u"[*]",u"", text)
    text = re.sub(u"\-+",u"-", text)
    text = re.sub(u'^ ',u'', text)
    text = re.sub(u'\u00E2',u'', text)
    text = re.sub(u'\u00E0',u'a', text)
    text = re.sub(u'\u00E9',u'e', text)
    
    return text

def sent2tokens(sent, language, lower=True):
    '''
    Sentence to stemmed tokens
    Parameter arguments:
    words = list of words e.g. sent = '... The boy is playing.'
    
    return:
    list of tokens
    ['the', 'boy', 'is', 'playing','.']
    '''
    if lower == True:
        sent = sent.lower()
    sent = text_normalization(sent)
    words = word_tokenize(sent, language)
    return words

def sent2stokens(sent, stemmer, language, lower=True):
    '''
    Sentence to stemmed tokens
    Parameter arguments:
    sent = a unicode string e.g. sent = '... The boys are playing'
    
    return:
    list of stemmed tokens
    ['the', 'boy', 'are', 'play', '.']
    '''
    words = sent2tokens(sent, language, lower)
    return [stemmer.stem(word) for word in words if not word.startswith(PUNCT)]

def remove_stopwords(words, stoplist):
    ''' Remove stop words
    Parameter arguments:
    words = list of words e.g. ['.', 'The', 'boy', 'is', 'playing', '.']
    
    return:
    list of tokens
    ['boy', 'is', 'playing']
    '''
    return [ token for token in words if not (token.startswith(PUNCT) or token in stoplist)]
    
def sent2tokens_wostop(sent, stoplist, language):
    '''
    Sentence to tokens without stopwords
    Parameter arguments:
    sent = a unicode string e.g. sent = '... The boys are playing'
    
    return:
    list of stemmed tokens without stop words
    ['boys', 'are', 'playing']
    '''
    
    words = sent2tokens(sent, language)    
    tokens = remove_stopwords(words, stoplist)
    return tokens

def sent2stokens_wostop(sent, stemmer, stoplist, language):
    '''
    Sentence to stemmed tokens without stopwords
    Parameter arguments:
    sent = a unicode string e.g. sent = '... The boys are playing'
    
    return:
    list of stemmed tokens without stop words
    ['boy', 'are', 'play']
    '''
    tokens = sent2tokens_wostop(sent, stoplist, language)
    return [stemmer.stem(token) for token in tokens]                            

def extract_ngrams(sentences, stoplist, stemmer, language, n=2):
    """Extract the ngrams of words from the input sentences.

    Args:
        n (int): the number of words for ngrams, defaults to 2
    """
    concepts = []
    for i, sentence in enumerate(sentences):

        # for each ngram of words
        tokens = sent2stokens_wostop(sentence, stoplist, language)
        for j in range(len(tokens)-(n-1)):

            # initialize ngram container
            ngram = []

            # for each token of the ngram
            for k in range(j, j+n):
                ngram.append(tokens[k].lower())

            # do not consider ngrams containing punctuation marks
            marks = [t for t in ngram if not re.search('[a-zA-Z0-9]', t)]
            if len(marks) > 0:
                continue

            # do not consider ngrams composed of only stopwords
            #stops = [t for t in ngram if t in stoplist]
            #if len(stops) == len(ngram):
                #continue

            # stem the ngram
            ngram = [stemmer.stem(t) for t in ngram]

            # add the ngram to the concepts
            concepts.append(' '.join(ngram))
    return concepts


def extract_ngrams2(sentences, stemmer, language, N=2):
    '''
    Parameter Arguments:
    sentences: list of sentences
             ['Ney York is a city.', 'It has a huge population.']
    N: Length of the n-grams e.g. 1, 2
    
    return: a list of n-grams
    [('new', 'york'), ('york', 'is'), ('is', 'a'), ('a', 'city'), (city, '.'), 
    ('it', 'has'), ('has','a'), ('a', 'huge'), ('huge', 'population') , ('population', '.')]
    '''
    ngrams_list = []
    for sent in sentences:
        sent = re.sub('[-](,?\s)','\\1', sent) #case where magister- has to be handled
        ngram_items = list(ngrams(sent2stokens(sent, stemmer, language), N))
        for i, ngram in enumerate(ngram_items):
            ngram_str = ' '.join(ngram)
          
            ngrams_list.append(ngram_str)
    return ngrams_list

def getTopNgrams(sentences, stemmer, language, stoplist, N=2, top=100):
    '''
    YG:
    get the top n-grams from the input sentences
    :param sentences:
    :param stemmer:
    :param language:
    :param stoplist:
    :param N:
    :param top:
    :return: a list of n-grams, like: ['New York', 'York is', ...]
    '''
    ngram_list = extract_ngrams_count(sentences, stemmer, language, stoplist, N)
    top_list = []
    while len(top_list) < top:
        highest_count = -1
        ngram = ''
        for key in ngram_list:
            if(ngram_list[key] > highest_count):
                highest_count = ngram_list[key]
                ngram = key
        top_list.append(ngram)
        del ngram_list[ngram]
    return top_list


def extract_ngrams_count(sentences, stemmer, language, stoplist, N=2):
    '''
    YG:
    extract n-grams and count the appearance times of each n-gram
    :param sentences: the list of sentences, each sentence is a string
    :param stemmer:
    :param language:
    :param N:
    :return:

    example input : 'This is a foo bar sentence'
    output: {'this is' : 1, 'is a' : 1, 'a foo' : 1, ...}
    the output is a dictionary
    '''
    #TODO: I am not sure whether we should remove all stopwords or not; maybe try both settings
    ngrams_count_dic= {}
    for i, sentence in enumerate(sentences):

        # for each ngram of words
        #sent = re.sub('[-](,?\s)','\\1', sentence) #case where magister- has to be handled
        #tokens = sent2stokens_wostop(sentence,stemmer,stoplist,language)
        tokens = sent2stokens(sentence,stemmer,language)
        for j in range(len(tokens)-(N-1)):
            # initialize ngram container
            ngram = []

            # for each token of the ngram
            for k in range(j, j+N):
                ngram.append(tokens[k].lower())

            # do not consider ngrams containing punctuation marks
            marks = [t for t in ngram if not re.search('[a-zA-Z0-9]', t)]
            if len(marks) > 0:
                continue

            # do not consider ngrams composed of only stopwords
            stops = [t for t in ngram if t in stoplist]
            if len(stops) == len(ngram):
                continue

            # stem the ngram
            #ngram = [stemmer.stem(t) for t in ngram]
            ngram = ' '.join(ngram)
            #print('ngram: '+repr(ngram))

            # add check whether this n-gram has already been contained in the n-grams list
            if ngram in ngrams_count_dic:
                ngrams_count_dic[ngram] = ngrams_count_dic[ngram] + 1
            else:
                ngrams_count_dic[ngram] = 1
    return ngrams_count_dic



def extract_entity_names(t):
    entity_names = []
    if hasattr(t, 'label'):
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))
    return entity_names

def get_phrases(sentence, phrase_type, language):
    tokens = sent2stokens(sentence, language, lower='False')
    tagged_sentence = pos_tag(tokens)
    tags = [tag for _, tag in tagged_sentence if re.match(r'NN.*|V.*|RB|JJ', tag)]

    phrases = []
    if phrase_type == 'entities':
        chunked_sentence = ne_chunk_sents([tagged_sentence], binary=True)
        for tree in chunked_sentence:
            phrases.extend(extract_entity_names(tree))    
    return phrases

def extract_nuggets(sentences, nugget_type, language):
    '''
    Parameter Arguments:
    sentences: list of sentences
             ['Ney York is a city.', 'It has a huge population.']
    
    return: a list of noun phrases, events, named_entities
    [('new', 'york'), ('york', 'is'), ('a', 'city'), 
    ('it', 'has'), ('has','a'), ('a', 'huge'), ('huge', 'population') , ('population', '.')]
    '''
    nugget_list = []
    for sent in sentences:
        if nugget_type == 'n-grams':
            nugget_items = list(ngrams(sent2stokens(sent, language), 2))
        if nugget_type == 'NP':
            nugget_items = get_phrases(sent, 'NP')
        if nugget_type == 'Phrases':
            nugget_items = get_phrases(sent, 'Phrases')
        if nugget_type == 'NE':
            nugget_items = get_phrases(sent, 'NE')
        for nugget in nugget_items:
            nugget_list.append(' '.join(nugget))
    return nugget_list

def prune_ngrams(ngrams, stoplist, N=2):
    pruned_list = []
    for ngram in ngrams:
        items = ngram.split(' ')
        i = 0
        for item in items:
            if item in stoplist: i += 1
        if i < N:
            pruned_list.append(ngram)
    return pruned_list

def get_sorted(dictionary):
    '''
    Sort the dictionary
    '''
    return sorted(dictionary, key=lambda x: dictionary[x], reverse=True)

def untokenize(tokens):
    """Untokenizing a list of tokens. 

    Args:
        tokens (list of str): the list of tokens to untokenize.

    Returns:
        a string

    """
    text = u' '.join(tokens)
    text = re.sub(u"\s+", u" ", text.strip())
    text = re.sub(u" ('[a-z]) ", u"\g<1> ", text)
    text = re.sub(u" ([\.;,-]) ", u"\g<1> ", text)
    text = re.sub(u" ([\.;,-?!])$", u"\g<1>", text)
    text = re.sub(u" _ (.+) _ ", u" _\g<1>_ ", text)
    text = re.sub(u" \$ ([\d\.]+) ", u" $\g<1> ", text)
    text = text.replace(u" ' ", u"' ")
    text = re.sub(u"([\W\s])\( ", u"\g<1>(", text)
    text = re.sub(u" \)([\W\s])", u")\g<1>", text)
    text = text.replace(u"`` ", u"``")
    text = text.replace(u" ''", u"''")
    text = text.replace(u" n't", u"n't")
    text = re.sub(u'(^| )" ([^"]+) "( |$)', u'\g<1>"\g<2>"\g<3>', text)

    # times
    text = re.sub('(\d+) : (\d+ [ap]\.m\.)', '\g<1>:\g<2>', text)

    text = re.sub('^" ', '"', text)
    text = re.sub(' "$', '"', text)
    text = re.sub(u"\s+", u" ", text.strip())

    return text

'''
def get_parse_info(parsestr):
    phrases = []
    tokens = Tree.fromstring(parsestr).leaves()
    for i in Tree.fromstring(parsestr).subtrees():
        if re.match('NP|CNP', i.label()):
            if i.height() == 3:
                if len(i.leaves()) == 1:
                    for child in i:
                        if re.match('PRP.*|EX|WP.*', child.label()):
                            pass 
                        else:
                            phrases.append(' '.join(i.leaves()))
                else:
                    phrases.append(' '.join(i.leaves()))
        #if i.label().startswith('N'):
        #    if len(i.leaves()) == 1:
        #        phrases.append(' '.join(i.leaves()))
        if i.label().startswith('V'):
            if i.label().startswith('VP'):
                for child in i:
                    if len(child.leaves()) == 1:
                        phrases.append(' '.join(child.leaves()))
            if len(i.leaves()) == 1:
                phrases.append(' '.join(i.leaves()))
    return tokens, phrases
'''

def flatten_childtrees(trees):
    children = []
    for t in trees:
        if t.height() < 3:
            children.extend(t.label())
        elif t.height() == 3:
            children.append(Tree(t.label(), t.pos()))
        else:
            children.extend(flatten_childtrees([c for c in t]))
    return children

def flatten_deeptree(tree):
    return Tree(tree.label(), flatten_childtrees([c for c in tree]))

def prune_phrases(phrases, stoplist, stemmer, language):
    pruned_list = []
    phrases = sorted(phrases, key=len, reverse=True)
    for phrase in phrases:
        tokens = sent2stokens(phrase, stemmer, language)
        ph = u' '.join(tokens)
        flag = 0
        for i, x in enumerate(pruned_list):
            if re.search(ph, x):
                flag = 1 
                break
        if ph in stoplist or flag == 1:
            continue
        else:
            pruned_list.append(ph)
    return pruned_list


def load_w2v_embeddings(embeddings_path, language, oracle_type):
    embeddings = {}
    if oracle_type.startswith('active_learning'):
        if language == 'english':
            #embeddPath = path.normpath(path.join(embeddings_path, "english/GoogleNews-vectors-negative300.bin.gz"))
            embeddPath = path.normpath(path.join(embeddings_path, "english/glove.6B.300d.txt"))
            embeddData = path.normpath(path.join(embeddings_path, "english/data/"))
            if not path.exists(embeddData):
                os.makedirs(embeddData)
            embedding_size = 300
            vocab_size = 400000 # Glove vectors
            binary_val = False
            embeddings = LoadEmbeddings(filepath=embeddPath, data_path=embeddData, vocab_size=vocab_size,
                                    embedding_size=embedding_size, binary_val=binary_val)
            
        if language == 'german':
            embeddPath = path.normpath(path.join(embeddings_path, "german/2014_tudarmstadt_german_50mincount.vec"))
            embeddData = path.normpath(path.join(embeddings_path, "german/data/"))
            vocab_size = 648460
            embedding_size = 100
            embeddings = LoadEmbeddings(filepath=embeddPath, data_path=embeddData, vocab_size=vocab_size,
                                    embedding_size=embedding_size)
    return embeddings