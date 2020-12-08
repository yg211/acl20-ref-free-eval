class Sentence:
    """The sentence data structure.

    Args: 
        tokens (list of str): the list of word tokens.
        doc_id (str): the identifier of the document from which the sentence
          comes from.
        position (int): the position of the sentence in the source document.
    """
    def __init__(self, tokens, doc_id, position):

        self.tokens = tokens
        """ tokens as a list. """

        self.doc_id = doc_id
        """ document identifier of the sentence. """

        self.position = position
        """ position of the sentence within the document. """

        self.concepts = []
        """ concepts of the sentence. """

        self.untokenized_form = ''
        """ untokenized form of the sentence. """

        self.length = 0
        """ length of the untokenized sentence. """