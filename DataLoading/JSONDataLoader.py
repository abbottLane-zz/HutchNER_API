# Copyright (c) 2016-2017 Fred Hutchinson Cancer Research Center
#
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#
from DataLoading.AbstractClasses import AbstractDataLoader
from NERPreprocessing.DocumentPreprocessing import UnformattedDocumentPreprocessor

from DataLoading.DataClasses import Document


class JSONDataLoader(AbstractDataLoader):
    def __init__(self,documents):
        super(JSONDataLoader, self).__init__()
        self.documents= documents

    def get_annotations(self):
        '''
        Public-pacing annotation getter. This should be used as info only, not for processing, as annotation formats
        differ depending on the source so we cannot garentee any standard format at tis point.

        The document objects retrieved by get_docs() will contain standardized annotations for downstream
        processing.
        :return: A list of dictionaries of {attrib_type<string>:value<string|int>}
        '''
        raise NotImplementedError("The JSON data loader does not load annotations: It can only consume a list of document text in JSON format")

    def preprocess(self, spacy_model):
        '''
        Using attributes from self, loads documents from JSON into memory (dict{doc_id:doc_text, ...:...})
        :return: True if load() succeeded, False otherwise
        '''
        doc_objs = dict()
        for id, doc in self.documents.items():
            doc_objs[id] = Document(id, doc)
        #Run loaded documents through preprocessor to get sentence segmentation, indx alignment, etc
        UnformattedDocumentPreprocessor(doc_objs, spacy_model=spacy_model)
        self.documents = doc_objs
        return self.documents


    def load_documents(self):
        '''
        Loads just the documents from LabKey server
        :return: list of Document objects
        '''
        raise NotImplementedError

    def load_annotations(self):
        '''
        Loads just the annotations from Labkey server
        :return: List of GoldAnnotation objects
        '''
        raise NotImplementedError("The JSON data loader does not load annotations: It can only read JSON documents in the bioNLP format")
