import re
from DataLoading.DataClasses import Sentence


class DocumentPreprocessor(object):
    """ NERPreprocessing parent object. Encapsulates all the preprocessing of loaded documents necessary for NER
    processing. This includes Sentence segmentation, Tokenization, POS tagging, SpaCy NER, Shape extraction,
    etc
    Args:
        documents_info (TextDataLoader) - Dataloading object for text/tsv files. An instantiated object of this
    type contains all the information about the documents that we need.
    """
    def __init__(self, documents, spacy_model):
        self.documents=documents
        self.nlppp = spacy_model
        print("Processing documents...")

    def _text2parseddata(self, doc):
        """
        Takes a document object, processes its sentences and populates object fields with preprocessing information
        :param doc: The document object which you'd like to put through the preprocessing pipline
        :return: A list of sentence objects which have been preprocessed
        """
        pass

    def get_processed_docs_and_info(self):
        """
        Returns a TextDataLoader object containing preprocessed document objects, and information bout the data
        Return:
            (TextDataLoader) object containing preprocessed document objects, and information bout the data
        """
        return self.documents


class i2b2DocumentPreprocessor(DocumentPreprocessor):
    """ Ecapsulates the preprocessing pipeline specific to data sources in the i2b2 format:
            - IE: Every line is a list of space seperated tokens representing a single sentence
    """
    def __init__(self, documents, spacy_model):
        super(i2b2DocumentPreprocessor, self).__init__(documents, spacy_model)
        docnum=0
        for docid, doc in self.documents.items():
            docnum+=1
            doc.sentences = self._text2parseddata(doc)
            if docnum%100 ==0 or docnum >= len(self.documents.keys()):
                print 'Processed %d documents out of %d' % (docnum, len(self.documents.keys()))
            doc.is_i2b2 = True
        print("Finished preprocessing documents.")

    def _text2parseddata(self, doc):
        """
        Given an document object, parse its text into sentence objects and set them in the Document object.
        Also populate the document's token list.
        :param doc: A Document object containing the document text.
        :return: The Sentence objects derived from the document text
        """
        sentences = [x for x in re.findall('(.*\n*)', doc.text)]
        sent_objs = list()
        begin = 0
        end = 0

        if sentences[-1] == "":
            del sentences[-1] # get rid of meaningless trailing tokens

        for index,sent_text in enumerate(sentences):
            if len(sent_text) == 0:
                sent_text = "\n"
            parsedData = self.nlppp(sent_text.decode("utf-8"))
            # update token spans
            updated_tok_spans = self._update_token_spans(begin, parsedData)
            doc.token_spans.extend(updated_tok_spans)
            sent_tokens = [x for x in parsedData]
            doc.tokens.extend(sent_tokens)
            last_token = parsedData[-1]
            len_last_token = len(last_token)
            last_token_idx = last_token.idx
            end = len_last_token + last_token_idx + begin
            sent_obj = Sentence(sent_text, begin, end, sent_tokens)
            sent_objs.append(sent_obj)
            begin = end
            if begin < len(doc.text):
                while doc.text[begin] == "\n" or doc.text[begin] == " " and begin < len(doc.text):
                    begin +=1 # compensate for any floating whitespace implicitly removed in tokenization
                    if begin >= len(doc.text):
                        break
        return sent_objs

    def _update_token_spans(self, begin, parsed_data):
        idxs = list()
        for x in parsed_data:
            idx = x.idx + begin
            idxs.append((idx, idx+len(x.orth_)))
        return idxs

class UnformattedDocumentPreprocessor(DocumentPreprocessor):
    """ Ecapsulates the preprocessing pipeline specific to un-pre-formatted data sources (ie not i2b2).
     Just raw text.
     """
    def __init__(self, documents, spacy_model):
        super(UnformattedDocumentPreprocessor, self).__init__(documents, spacy_model)
        dnum = 0
        for docid, doc in self.documents.items():
            dnum += 1
            if dnum % 100 == 0: # print status of every 100 docs to keep user updated
                print "Document pre-processing on doc " + str(dnum) + "/" + str(
                    len(self.documents))
            doc.sentences = self._text2parseddata(doc)
        print("Finished pre-processing documents.")

    def _text2parseddata(self, document):
        """
        Given an document object, parse its text into sentence objects and set them in the Document object.
        Also populate the document's token list.
        :param doc: A Document object containing the document text.
        :return: The Sentence objects derived from the document text
        """
        text = document.text
        # parse the data
        parsedData = self.nlppp(unicode(text))
        # store document tokens
        document.tokens = [x for x in parsedData]
        document.token_spans = [(x.idx, x.idx + len(x.orth_)) for x in parsedData]
        # Extract the various sentence representations
        sents = []
        for span in parsedData.sents:
            senttext = span.text
            spanbegin = span.start_char
            spanend = span.end_char
            tokens = [parsedData[i] for i in range(span.start, span.end)]

            sent = Sentence(senttext, spanbegin, spanend, tokens)
            sents.append(sent)
        return sents
