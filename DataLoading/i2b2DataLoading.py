import os

from HutchNER.NERPreprocessing.DocumentPreprocessing import i2b2DocumentPreprocessor

from DataClasses import GoldAnnotation
from HutchNER.DataLoading.TextDataLoading import TextDataLoader


class i2b2DataLoader(TextDataLoader):
    def __init__(self, txt_dir, annotation_dir = None):
        super(i2b2DataLoader, self).__init__(txt_dir)
        self.annotation_dir = annotation_dir
        self.detected_labels = set()

    def load(self):
        '''
        Using attributes from self, loads documents and annotations in i2b2 format from local dirs into memory
        :return: Document Objs 
        '''
        docs = self.load_documents()

        # Sentence segmentation, tokenization, POS, dep parsing, etc (req'd before adding annotations)
        i2b2DocumentPreprocessor(docs)

        # Add annotations to document objects
        if self.annotation_dir:
            self.join_annotations(docs)
        return docs

    def get_annotations(self):
        '''
        Public-pacing annotation getter. This should be used as info only, not for processing, as annotation formats
        differ depending on the source so we cannot garentee any standard format at tis point.

        The document objects retrieved by get_docs() will contain standardized annotations for downstream
        processing.
        :return: A list of dictionaries of {attrib_type<string>:value<string|int>}
        '''
        if self.annotations:
            return self.annotations
        else:
            raise ValueError("There were no annotations retrieved in dataloading, so 'get_annotations()' returns nothing")

    def _get_annotations(self):
        '''
        Loads just the annotations from Labkey server
        :return: List of GoldAnnotation objects
        '''
        document_sentidx_data = dict()
        for filename in os.listdir(self.annotation_dir):
            with open(os.path.join(self.annotation_dir, filename), "rb") as f:
                annot_lines = f.readlines()
                doc_id = filename.split(".")[0]

                if doc_id not in document_sentidx_data:
                    document_sentidx_data[doc_id] = list()
                for line in annot_lines:
                    annot_tuple = self._parse_i2b2_annotation(line)
                    sent_idx = int(annot_tuple[0]) -1
                    tok_begin = annot_tuple[1]
                    tok_end = annot_tuple[2]
                    tag = annot_tuple[3].lower()
                    self.detected_labels.add(tag)
                    document_sentidx_data[doc_id].append((sent_idx, tok_begin, tok_end, tag))
        return document_sentidx_data

    def join_annotations(self, docs):
        self.docs = docs
        if self.annotation_dir:
            self.annotations = self._get_annotations()
            # Combine docs and annotations here
            for doc_id, anns in self.annotations.items():
                for a in anns:
                    self.add_annotation(doc_id, a)
        return self.docs

    def _parse_i2b2_annotation(self, line):
        concept = line.split("||")

        iob_wordIdx = concept[0].split()
        # print concept[0]
        iob_class = concept[1].split("=")
        iob_class = iob_class[1].replace("\"", "")
        iob_class = iob_class.replace("\n", "")

        # print iob_wordIdx[len(iob_wordIdx)-2],iob_wordIdx[len(iob_wordIdx)-1]
        start_iobLineNo = iob_wordIdx[len(iob_wordIdx) - 2].split(":")
        end_iobLineNo = iob_wordIdx[len(iob_wordIdx) - 1].split(":")
        start_idx = start_iobLineNo[1]
        end_idx = end_iobLineNo[1]
        iobLineNo = start_iobLineNo[0]

        return (iobLineNo, start_idx, end_idx, iob_class)

    def add_annotation(self, doc_id, annotation):
        doc = self.docs[doc_id]
        sent_idx = annotation[0]
        start_tok = annotation[1]
        end_tok = annotation[2]
        start_idx, end_idx, text = self._tokspan2docspan_simple(doc, sent_idx, start_tok, end_tok)
        tag = annotation[3]
        e = GoldAnnotation(tag, start_idx, end_idx, text, sent_idx)
        if tag not in doc.concepts_gold:
            doc.concepts_gold[tag] = list()
        doc.concepts_gold[tag].append(e)
        return True

    def _tokspan2docspan_simple(self, doc, sentidx, start_tok, end_tok):
        sent = doc.sentences[sentidx]
        sent_text = sent.text
        i2b2_tokens = sent.text.split()
        idexes_of_space = self._find_tok_boundaries(sent_text, delimiter=' ')
        sent_begin, sent_end = self._get_spans_from_tokens(idexes_of_space, i2b2_tokens, start_tok, end_tok)
        doc_begin = sent_begin + doc.sentences[sentidx].span_start
        doc_end = sent_end + doc.sentences[sentidx].span_start
        text = doc.text[doc_begin:doc_end]
        return doc_begin, doc_end, text

    def _get_spans_from_tokens(self, space_indexes, i2b2_tokens, starttok, endtok):
        start_buffer = 0 if int(starttok) == 0 else 1
        end_buffer = 0 if int(endtok) == 0 else 1
        try:
            sent_begin = space_indexes[int(starttok)] + start_buffer
        except:
            pause=1
        sent_end = space_indexes[int(endtok)] + end_buffer + len(i2b2_tokens[int(endtok)])
        return sent_begin, sent_end

    def _find_tok_boundaries(self, s, delimiter):
        spans = list()
        spans.append(0)
        spans.extend([i for i, ltr in enumerate(s) if ltr == delimiter])
        spans.append(len(s))
        return spans

