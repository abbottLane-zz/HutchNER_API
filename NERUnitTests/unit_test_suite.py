import unittest

from collections import defaultdict

import en_core_web_sm
from NERExtraction.Extraction import NERExtraction

from DataLoading.DataClasses import Document
from DataLoading.LabKeyDataLoading import LabKeyDataLoader
from DataLoading.SectionerXMLDataLoading import SectionerXMLDataLoader
from DataLoading.TextDataLoading import TextDataLoader
from DataLoading.i2b2DataLoading import i2b2DataLoader
from NERExtraction import FeatureProcessing
from NERNegation.NegEx.HutchNegEx import HutchNegEx
from NERPreprocessing.DocumentPreprocessing import i2b2DocumentPreprocessor, UnformattedDocumentPreprocessor
from NERUtilities.MiscFunctions import load_labkey_server_info_from_ini
# initialize spacy for preprocessing
spacy_model = en_core_web_sm.load()


class DocumentTestCase(unittest.TestCase):
    def setUp(self):
        self.document = Document("PID1234_DID1234", "This is the text of the document.\nIt has many words. "
                                                       "Tremendous words.\nThe patient has a 500 year smoking "
                                                       "history.\n\tSAD.")

    def tearDown(self):
        self.document = None

    def test_document_default_values(self):
        self.assertEqual(self.document.document_id, "PID1234_DID1234")
        self.assertEqual(self.document.patient_id, "PID1234")
        self.assertEqual(self.document.text, "This is the text of the document.\nIt has many words. "
                                                       "Tremendous words.\nThe patient has a 500 year smoking "
                                                       "history.\n\tSAD.")


class TextDataLoaderTests(unittest.TestCase):
    def setUp(self):
        # load raw text
        text_dir = "/home/wlane/PycharmProjects/HutchNER_API/NERResources/TestCaseData_i2b2"
        self.text_dl = TextDataLoader(text_dir)
        docs = self.text_dl.load()
        UnformattedDocumentPreprocessor(docs, spacy_model)
        self.docs = docs

    def tearDown(self):
        self.text_dl = None

    def test_TextDataLoader_default_values(self):
        documents = self.docs
        self.assertEqual(len(documents), 6)

        for id, text in documents.items():
            self.assertEqual(len(text.token_spans), len(text.tokens), "Token spans list is not the same length as token list")

            for sent in text.sentences:
                self.assertEqual(len(sent.dependency_tags),
                                 len(sent.dependency_tags_heads))
                self.assertEqual(len(sent.entity_label),
                                 len(sent.pos_tags))
                self.assertEqual(len(sent.shape_tokens),
                                 len(sent.pos_tags))
                self.assertEqual(sent.span_end-sent.span_start, len(sent.text),
                                 msg="Sentence spans don't match sentence length: \n\tspan_start:"
                                 ""+str(sent.span_start) + " span_end:"+str(sent.span_end) + " Diff:" +
                                     str(sent.span_end-sent.span_start)+"\n\tLength:"
                                     + str(len(sent.text)))


class LabkeyDataLoaders(unittest.TestCase):
    def setUp(self):
        # Load local i2b2 data
        data_source_info = load_labkey_server_info_from_ini("SocHistLabkeyTest")
        self.lk_dl = LabKeyDataLoader(data_source_info["driver"], data_source_info["database"],
                                 data_source_info["server"], data_source_info["table"],
                                 data_source_info["job_run_ids"])
        self.lk_dl.load()

    def tearDown(self):
        self.lk_dl = None

    def test_LKDataLoader_default_values(self):
        documents = self.lk_dl.load_documents()
        self.assertEqual(len(documents), 0, msg="Job run ID 1221 should have 0 approved documents")

        for id, text in documents.items():
            self.assertEqual(len(text.token_spans), len(text.tokens), "Token spans list is not the same length as token list")
            self.assertNotEqual(None, text.patient_id, msg="All documents from LabKey should have a valid patient_id")
            self.assertNotEqual(None, text.document_id, msg="All documents from labkey should have a valid document_id")

            for sent in text.sentences:
                self.assertEqual(len(sent.dependency_tags),
                                 len(sent.dependency_tags_heads), msg="Length of a sentence's token-layer metadata should all be uniform. Problem: Dependency tags, Dependency tag heads")
                self.assertEqual(len(sent.entity_label),
                                 len(sent.pos_tags), msg="Length of a sentence's token-layer metadata should all be uniform. Problem: Entity Labels, POS tags")
                self.assertEqual(len(sent.shape_tokens),
                                 len(sent.pos_tags), msg="Length of a sentence's token-layer metadata should all be uniform. Problem: shape_tokens, pos_tags")
                self.assertEqual(sent.span_end-sent.span_start, len(sent.text),
                                 msg="Sentence spans don't match sentence length: \n\tspan_start:"
                                 ""+str(sent.span_start) + " span_end:"+str(sent.span_end) + " Diff:" +
                                     str(sent.span_end-sent.span_start)+"\n\tLength:"
                                     + str(len(sent.text)))


class i2b2DataLoaderTests(unittest.TestCase):
    def setUp(self):
        # Load local i2b2 data
        self.i2b2_txt = "/home/wlane/nethome/i2b2_data/2010_concepts_plusFH/train/txt/"
        self.i2b2_annotations = "/home/wlane/nethome/i2b2_data/2010_concepts_plusFH/train/concept/"

        self.i2b2_dl = i2b2DataLoader(annotation_dir=self.i2b2_annotations, txt_dir=self.i2b2_txt)
        self.i2b2_dl.load()

    def tearDown(self):
        self.i2b2_dl = None

    def test_i2b2DataLoader_default_values(self):
        documents = self.i2b2_dl.load()

        for id, text in documents.items():
            self.assertEqual(len(text.token_spans), len(text.tokens), "Token spans list is not the same length as token list")
            self.assertNotEqual(None, text.document_id, msg="All documents from i2b2 should have a valid document_id")

            for sent in text.sentences:
                self.assertEqual(len(sent.dependency_tags),
                                 len(sent.dependency_tags_heads), msg="Length of a sentence's token-layer metadata should all be uniform. Problem: Dependency tags, Dependency tag heads")
                self.assertEqual(len(sent.entity_label),
                                 len(sent.pos_tags), msg="Length of a sentence's token-layer metadata should all be uniform. Problem: Entity Labels, POS tags")
                self.assertEqual(len(sent.shape_tokens),
                                 len(sent.pos_tags), msg="Length of a sentence's token-layer metadata should all be uniform. Problem: shape_tokens, pos_tags")
                self.assertEqual(sent.span_end-sent.span_start, len(sent.text),
                                     msg="Sentence spans don't match sentence length: \n\tspan_start:"
                                     ""+str(sent.span_start) + " span_end:"+str(sent.span_end) + " Diff:" +
                                         str(sent.span_end-sent.span_start)+"\n\tLength:"
                                         + str(len(sent.text)) + " Text: " + sent.text)


class i2b2PreprocessorTests(unittest.TestCase):
    def setUp(self):
        # Load local i2b2 data
        i2b2_txt = "/home/wlane/nethome/i2b2_data/2010_concepts_plusFH/train/txt/"
        i2b2_annotations = "/home/wlane/nethome/i2b2_data/2010_concepts_plusFH/train/concept/"

        self.i2b2_dl = i2b2DataLoader(annotation_dir=i2b2_annotations, txt_dir=i2b2_txt)
        self.documents =self.i2b2_dl.load()
        self.preprocess_eng = i2b2DocumentPreprocessor(self.documents, spacy_model)

    def tearDown(self):
        self.preprocess_eng = None
        self.documents = None
        self.i2b2_dl = None


    def test_text2parseddata_simple(self):
        doc = Document("pid123_12432", "This is a Document .\nThere are multiple sentences .\nThree , to be exact .")

        sentences= self.preprocess_eng._text2parseddata(doc)
        self.assertEqual(len(sentences), 3)
        # Sentence 0:
        self.assertEqual(sentences[0].text, "This is a Document .\n")
        self.assertEqual(sentences[0].span_start, 0)
        self.assertEqual(sentences[0].span_end, 21)
        self.assertEqual(doc.text[sentences[0].span_start:sentences[0].span_end], sentences[0].text)
        # sentence 1:
        self.assertEqual(sentences[1].text, "There are multiple sentences .\n")
        self.assertEqual(sentences[1].span_start, 21)
        self.assertEqual(sentences[1].span_end, 52)
        self.assertEqual(doc.text[sentences[1].span_start:sentences[1].span_end], sentences[1].text)


class FeatureGenerationTests(unittest.TestCase):
    def setUp(self):
        self.doc1 = Document("pid123_12432", "This is a Document .\nThere are multiple sentences .\nThree , to be exact .")
        self.doc2 = Document("11232321_3433", "The pt complains of fever , headaches , nightsweats .\n Prescription of a super medicine was given . Tests :\n blood : normal , oxygen : 1231/1242 , tox : b7+")
        self.doc3 = Document("545674_3234", "Myaloma was detected .\nThere are multiple sarcoma , psychosis .")
        self.doc4 = Document("235235_32", "Surgeyr was schduled for next wednesday , but pt did not show up .\nDr. Joe called him and asked what's the deal bruh .\nPatient had overslept .\n")
        self.doc5 = Document("8563234_44", "Path report showed malignant status of non-small cell tumor in lung .\nNo clue what that means but it sounds serious .\nc4N3n2 sometimes garbage : gets into a sentence saf ::fsd .")

        self.docs = {self.doc1.document_id : self.doc1,
                self.doc2.document_id : self.doc2,
                self.doc3.document_id : self.doc3,
                self.doc4.document_id : self.doc4,
                self.doc5.document_id : self.doc5}

        self.preprocess_eng = i2b2DocumentPreprocessor(self.docs, spacy_model)
        self.docs = self.preprocess_eng.get_processed_docs_and_info()

    def tearDown(self):
        self.docs = None
        self.preprocess_eng = None

    def test_features_doc3(self):
        for i, s in enumerate(self.doc3.sentences):
            sent_toks = s.tokens
            feature_vecs = FeatureProcessing.sent2features(sent_toks)
            if i==0:
                self.assertEqual(feature_vecs[0][3], "word.lower=myaloma")
                self.assertEqual(feature_vecs[0][12], "hasProblemForm=True")
                self.assertEqual(feature_vecs[0][13], "BOS")
                self.assertEqual(feature_vecs[3][20], "+1:featureMetricUnit=None")

            elif i==1:
                self.assertEqual(feature_vecs[3][3], "word.lower=sarcoma")
                self.assertEqual(feature_vecs[3][12], "hasProblemForm=True")
                self.assertEqual(feature_vecs[5][3], "word.lower=psychosis")
                self.assertEqual(feature_vecs[5][12], "hasProblemForm=True")
                self.assertEqual(feature_vecs[6][20], "EOS")

    def eos_bos_placement(self):
        for i, doc in enumerate(self.docs):
            for s in doc.sentences:
                sent_toks = s.tokens
                feature_vecs = FeatureProcessing.sent2features(sent_toks)
                eos_pos = 20
                bos_pos = 13
                for tok_num, token_feats in enumerate(feature_vecs):
                    for feat in token_feats:
                        if feat =="BOS":
                            self.assertEqual(bos_pos, tok_num)
                        elif feat =="EOS":
                            self.assertEqual(eos_pos,tok_num)


class XMLDataLoaderTests(unittest.TestCase):
    def setUp(self):
        self.dl = SectionerXMLDataLoader("/home/wlane/PycharmProjects/HutchNER_API/NERResources/TestCaseData_sectioned", clean_tmp_files=False)

    def tearDown(self):
        self.dl=None

    def test_get_doc_id(self):
        did1="12432342_23424"
        did2="1243432_21312.txt"
        did3="this.is.a.doc.id.xml"

        self.assertEqual(self.dl._get_doc_id(did1), "12432342_23424")
        self.assertEqual(self.dl._get_doc_id(did2), "1243432_21312")
        self.assertEqual(self.dl._get_doc_id(did3), "this.is.a.doc.id")

    def standard_case_load(self):
        loaded_docs = self.dl.preprocess(spacy_model=spacy_model)
        self.assertEqual(len(loaded_docs), 3)
        self.assertEqual(loaded_docs[0].document_id, 'NERTraining.b0.doc13')
        self.assertEqual(loaded_docs[1].document_id, 'NERTraining.b0.doc15')
        self.assertEqual(loaded_docs[2].document_id, 'NERTraining.b0.doc14')

    def test_get_section_names(self):
        loaded_docs = self.dl.preprocess(spacy_model=spacy_model)

        s1 = loaded_docs['NERTraining.b0.doc13'].get_detected_section_names()
        s2 = loaded_docs['NERTraining.b0.doc14'].get_detected_section_names()
        s3 = loaded_docs['NERTraining.b0.doc15'].get_detected_section_names()

        self.assertEqual(set(s1), {'Past Surgical History', 'Allergies', 'Past Medical History', 'Family History',
                                   'Social History', 'History', 'Physical'})
        self.assertEqual(set(s2), {'Allergies', 'Social History'})
        self.assertEqual(set(s3), set([]))

    def test_get_section_tokens(self):
        loaded_docs = self.dl.preprocess(spacy_model=spacy_model)
        UnformattedDocumentPreprocessor(loaded_docs, spacy_model=spacy_model)
        tester = NERExtraction(loaded_docs)
        loaded_docs = tester.tag_all()

        # standard use case
        surg_history_section = loaded_docs['NERTraining.b0.doc13'].get_section_tokens(['Past Surgical History'])
        surg_and_soc_history_sections = loaded_docs['NERTraining.b0.doc13'].get_section_tokens(['Past Surgical History', 'Social History'])
        allergies_and_soc_history = loaded_docs['NERTraining.b0.doc14'].get_section_tokens(['Allergies', 'Social History'])

        # testing method when section requested is not present
        allergies_and_soc_history = loaded_docs['NERTraining.b0.doc15'].get_section_tokens(['Allergies', 'Social History'])
        self.assertEqual(allergies_and_soc_history, defaultdict(list))


class negex_tests(unittest.TestCase):
    def setUp(self):
        self.negater = HutchNegEx()
        self.dl = SectionerXMLDataLoader("/home/wlane/PycharmProjects/HutchNER_API/NERResources/TestCaseData_sectioned", clean_tmp_files=False)
        loaded_docs = self.dl.preprocess(spacy_model=spacy_model)
        UnformattedDocumentPreprocessor(loaded_docs, spacy_model=spacy_model)
        tester = NERExtraction(loaded_docs)
        self.loaded_docs = tester.tag_all()


    def tearDown(self):
        self.dl=None
        self.loaded_docs=None
        self.negater=None

    def test_match_negations(self):
        # standard case
        text1 = "He is not on erlotinib."
        negations1 = self.negater._match_negation(text=text1)
        self.assertEqual(len(negations1), 1)
        self.assertEqual(negations1[0][0], (5,10))

        # case where last token is a negation trigger
        text2 = "erlotinib, he went without."
        negations2 = self.negater._match_negation(text=text2)
        self.assertEqual(len(negations2), 1)
        self.assertEqual(negations2[0][0], (18,27))

        # case with negated trigger is capitalized
        text3 = "erlotinib was NOT taken by the patient prior to surgery."
        negations3 = self.negater._match_negation(text=text3)
        self.assertEqual(len(negations3), 1)
        self.assertEqual(negations3[0][0], (13,18))

    def test_negate_named_entities(self):
        for id, doc in self.loaded_docs.items():
            self.negater.negate(doc)
        doc1 = self.loaded_docs['NERTraining.b0.doc13']
        doc2 = self.loaded_docs['NERTraining.b0.doc14']
        doc3 = self.loaded_docs['NERTraining.b0.doc15']

        self.assertEqual(doc3.NER_token_labels[118]['negation'], "AMBIVALENT_EXISTENCE")
        self.assertEqual(doc3.NER_token_labels[119]['negation'], "AMBIVALENT_EXISTENCE")


if __name__ == '__main__':
    unittest.main()