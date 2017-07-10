# Copyright (c) 2016-2017 Fred Hutchinson Cancer Research Center
#
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#
import csv
import os
from os.path import isfile, join

from DataLoading.AbstractClasses import AbstractDataLoader
from DataLoading.DataClasses import Document


class TextDataLoader(AbstractDataLoader):
    def __init__(self, txt_dir, annotation_dir=None):
        super(TextDataLoader, self).__init__()
        self.txt_dir = txt_dir
        self.annotation_dir = annotation_dir

    def load(self):
        '''
        Using attributes from self, loads documents from unformatted text in local dirs into memory.
        Unformatted text is, for the moment, assumed to not come with any annotation, presumed to be used in
        a simple tagging environment.
        :return: Document Objs
        '''
        return self.load_documents()

    def load_documents(self):
        '''
        Loads just the documents from LabKey server
        :return: dict of {DocId<string>: Document<object>, ...}
        '''
        documents = {}
        files_in_dir = [f for f in os.listdir(self.txt_dir) if isfile(join(self.txt_dir, f))]
        for f in files_in_dir:
            if f[-3:] == "tsv":
                # parse as a tsv or csv file
                documents.update(self._parse_tsv(join(self.txt_dir, f)))
            elif f[-3:] == "csv":
                raise ValueError("Document type \'csv\' incompatible with this engine. Feed me a tsv.")
            else:
                # parse as a text file
                documents.update(self._parse_text(join(self.txt_dir, f)))
        return documents

    def _parse_tsv(self, filepath):
        documents = dict()
        with open(filepath, "rb") as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar="\"")
            # Make a document object for each row
            for i, row in enumerate(reader):
                if i !=0: # First row is headers, not a real row
                    doc_id = row[1]
                    text = row[3]
                    if doc_id in documents:
                        print("Warning. Possible Document Duplicate found (same docID: " + doc_id + ") is not "
                                        "a unique value. Either you have duplicate documents, or your doc_id scheme is"
                                        " not thorough enough. Excluding document and continuing processing...")
                    else:
                        documents[doc_id] = Document(doc_id, text)
        return documents

    def _parse_text(self, filepath):
        documents = dict()
        # DocId is defined in a text document by the text's title
        with open(filepath, "rb") as f:
            text = f.read()
            doc_id = filepath.split(os.path.sep)[-1].split(".")[0]
            documents[doc_id]=Document(doc_id,text)
        return documents