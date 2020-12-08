import numpy as np
import os
import os.path as path
from gensim.models import KeyedVectors
import codecs

class LoadEmbeddings():   
    def __init__(self, filepath, data_path, vocab_size, embedding_size=300, binary_val=True):
        self.vocab_dict = {}
        self.embedding_size = embedding_size
        self.loadEmbeddings(filepath, data_path, vocab_size, binary_val)
       
    def convertToNumpy(self, vector):
        return np.array([float(x) for x in vector])

    def loadEmbeddings(self, filepath, data_path, vocab_size, binary_val):
        embed_short = os.path.normpath("%s/embed.dat" % data_path)
        if not os.path.exists(embed_short):
            print("Caching word embeddings in memmapped format...")
            print(binary_val, filepath)
            wv = KeyedVectors.load_word2vec_format("%s" % (filepath), binary=binary_val)
            fp = np.memmap(embed_short, dtype=np.double, mode='w+', shape=wv.syn0.shape)
            fp[:] = wv.syn0[:]
            with open(os.path.normpath("%s/embed.vocab" % data_path), "w") as fp:
                for _, w in sorted((voc.index, word) for word, voc in wv.vocab.items()):
                    fp.write("%s\n"%(w.encode("utf8")))
            del fp, wv
            
        self.W = np.memmap(os.path.normpath("%s/embed.dat" % data_path), dtype=np.double, mode="r", shape=(vocab_size, self.embedding_size))
        with codecs.open(os.path.normpath("%s/embed.vocab" % data_path), 'r', 'utf-8') as f:
            vocab_list = [x.strip() for x in f.readlines()]
        self.vocab_dict = {w: k for k, w in enumerate(vocab_list)}

    def word2embedd(self, word):
        word = word.lower()
        if word in self.vocab_dict:
            return self.W[self.vocab_dict[word]]
        else:
            return self.W[self.vocab_dict["unknown"]]
    
    def isKnown(self, word):
        word = word.lower()
        return word in self.vocab_dict
    
if __name__ == '__main__':
    language =  'german'
    data_path = os.path.normpath("%s/../data" % (path.dirname(path.abspath(__file__))))
    if language == 'english':
        embeddPath = os.path.normpath("%s/embeddings/english/GoogleNews-vectors-negative300.bin.gz" % (data_path))
        embeddData = os.path.normpath("%s/embeddings/english/data/" % (data_path))
        vocab_size = 3000000
        embedding_size = 300
    if language == 'german':
        embeddPath = os.path.normpath("%s/embeddings/german/2014_tudarmstadt_german_50mincount.vec" % (data_path))
        embeddData = os.path.normpath("%s/embeddings/german/data/" % (data_path))
        vocab_size = 648460
        embedding_size = 100
        
    embeddings = LoadEmbeddings(filepath=embeddPath, data_path=embeddData, vocab_size=vocab_size, embedding_size=embedding_size)
    print(embeddings.word2embedd("der"))
    print(embeddings.word2embedd("unknown"))