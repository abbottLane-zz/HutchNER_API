from os import listdir
import os
from os.path import isfile, join
import logging
from DataLoading.AbstractClasses import AbstractDataLoader
import xmltodict
from DataLoading.DataClasses import Document
from NERPreprocessing.DocumentPreprocessing import UnformattedDocumentPreprocessor


class SectionerXMLDataLoader(AbstractDataLoader):
    def __init__(self,xml_dir, clean_tmp_files=True):
        super(SectionerXMLDataLoader, self).__init__()
        self.xml_dir = xml_dir
        self.clean_tmp_files = clean_tmp_files

    def get_annotations(self):
        '''
        Public-pacing annotation getter. This should be used as info only, not for processing, as annotation formats
        differ depending on the source so we cannot garentee any standard format at tis point.

        The document objects retrieved by get_docs() will contain standardized annotations for downstream
        processing.
        :return: A list of dictionaries of {attrib_type<string>:value<string|int>}
        '''
        raise NotImplementedError("The XML data loader does not load annotations: It can only read XML documents in the bioNLP format")

    def preprocess(self, spacy_model):
        '''
        Using attributes from self, loads documents from XML into memory
        :return: True if load() succeeded, False otherwise
        '''
        onlyfiles = [f for f in listdir(self.xml_dir) if isfile(join(self.xml_dir, f))]
        all_docs = dict()
        for doc in onlyfiles:
            doc_id = self._get_doc_id(doc)
            try:
                with open(join(self.xml_dir, doc), "rb") as f:
                    xml_obj = xmltodict.parse(f.read())
                    full_text = xml_obj['AnnotatedDocument']['DocumentText']
                    annotation_sections = xml_obj['AnnotatedDocument']['TextAnnotationSet']['TextAnnotation']

                    all_sections = dict()
                    for section in annotation_sections:
                        current_section = {
                            'end':section['@End'],
                            'start': section['@Start'],
                            'type': section['@Type'],
                            'text': section['Text'],
                            'label':section['FeatureSet']['Feature']['#text']
                            }
                        if current_section['label'] not in all_sections:
                            all_sections[current_section['label']] = list()
                        all_sections[current_section['label']].append(current_section)
                    doc_obj = Document(doc_id, full_text)
                    doc_obj.set_sections(all_sections)
                    all_docs[doc_id]=doc_obj
            except Exception, e:
                logging.info("Failed to load XML sectioned documents for document_id: " + doc_id)
                doc_obj = Document(doc_id, full_text)
                all_docs[doc_id]=doc_obj

        #Run loaded documents through preprocessor to get sentence segmentation, indx alignment, etc
        UnformattedDocumentPreprocessor(all_docs, spacy_model=spacy_model)

        # clear out sectioned tmp folder if clean run
        if self.clean_tmp_files:
            self._clear_tmp_section_data()
        return all_docs


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
        raise NotImplementedError("The XML data loader does not load annotations: It can only read XML documents in the bioNLP format")

    def _get_doc_id(self, doc):
        items = doc.split('.')
        if len(items)==1:
            return items[0]
        else:
            return '.'.join(items[0:-1])

    def _clear_tmp_section_data(self):
        sectioner_dir = self.xml_dir

        for the_file in os.listdir(sectioner_dir):
            file_path = os.path.join(sectioner_dir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)
