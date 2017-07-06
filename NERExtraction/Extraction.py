# Copyright (c) 2016-2017 Fred Hutchinson Cancer Research Center
#
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#
import json
import os
from os.path import isfile, join

import re
from flask.json import jsonify
from sklearn.externals import joblib

from LSTMExec import predict_lstm
from NERNegation.NegEx.HutchNegEx import HutchNegEx

from FeatureProcessing import sent2features
from NERUtilities.Clusters import Clusters


class NERExtraction:
    """
    The object driving the Clinical concept extraction Testing pipeline
    """
    def __init__(self, documents, model_algo="crf"):
        self.model_algo = model_algo
        self.documents = documents
        self.clusters = Clusters(os.path.join(os.path.dirname(__file__),os.path.join("..","NERResources","Cluster_Files",
                                              "MMC867k_FH255k.600.cbow.model.bin_k=800minibatch=False.kmeans")))

        self.negexer = HutchNegEx()
        self.possible_labels = []
        self.model_paths_by_concepts= None
        self.model_dir = None

    def tag_all(self, models):
        """
        Initiates concept extraction testing pipeline over all files in self.data_dir
        :return: list of Document objects where Document.predicted has been populated with CRF predictions
        """
        if self.model_algo.lower() == "lstm": # Use the LSTM model and decoder
            docs = self.documents
            tags_and_toks_by_doc_id = predict_lstm.main(docs, models["lstm_crf"])
            docs = self._combine_docs_and_predictions(docs, tags_and_toks_by_doc_id)
            print ("Finished LSTM classification")
            return docs

        elif self.model_algo.lower() == "crf": # USe CRF model and decoder
            docs = self.documents
            self._extract(docs, models["crf_ner"], "crf_ner")
            print("Finished CRF classification")
            return docs

    def _extract(self, doc_objs_dict, model, model_name):
        print "Pulling out " + model_name + " information ..."
        self.possible_labels = list(model.classes_)
        self.possible_labels.remove("O")
        for i, current_doc in enumerate(doc_objs_dict.values()):
            # generate feature vectors
            feature_vectors = sent2features(current_doc.tokens, clusters=self.clusters)
            # Predict type sequence
            result_probabilities = model.predict_marginals_single(feature_vectors)
            # Set prediction in document object
            current_doc.set_NER_predictions(result_probabilities, model_name)

    def _print_marginal_sequences(self, tagger, predictions, label, tokens):
        for i, p in enumerate(predictions):
            if p != "O":
                print "Marginal for ["+tokens[i].string +"] as \'" + label + "\': " + str(tagger.marginal(label, i)) + " and for O: " + str(
                    tagger.marginal("O", i))

    def _modeldir2concepts(self, dir):
        """
        Takes a directory path to where all CRF models live, and extracts the name of the clinical concept from the name of the model, using
        that as a key and the filepath as the value
        :param dirs: (list of str) List of paths to model files
        :return: Dictionary of clinical concept strings mapped to the model path
        """
        dirs = [f for f in os.listdir(dir) if isfile(join(dir, f))]
        model_names = dict()
        for dir in dirs:
            try:
                # Assumes naming convention: /somepath/that/leads/to/model/model-{modelName}.ser.gz
                model_names[dir.split(os.sep)[-1].split('-')[1].split('.')[0]] = dir
            except:
                raise ValueError("Model file names don't conform to the naming convention: model-{nameOfConcept}.ser.gz: \n\t" + dir)
        return model_names

    def _combine_docs_and_predictions(self, docs, tags_n_toks):
        """
        Takes document info object and tags and tokens and puts them together in the correct format for the DocObj
        :param doc_info: An DataLoading object containing info about all documents
        :param tags_n_toks: A dictionary fo {docId:([token1Text, token2text, ...] , [label1Text, label2text, ...,])}
        :return: A list of Document objects, populated with prediction information
        """
        documents = docs
        alllabels = set()
        for docid, tags_tuple in tags_n_toks.items():
            tags = tags_tuple[0]
            toks = tags_tuple[1]
            taglist=[item for sublist in tags for item in sublist]
            tokslist = [item for sublist in toks for item in sublist]
            documents[docid].NER_token_labels = self._transform_to_dict(taglist, tokslist, documents[docid].token_spans)
            documents[docid].tokens = tokslist
            alllabels = alllabels.union(set(taglist))
        alllabels.remove("O")
        self.possible_labels = list(alllabels)
        return documents

    def _transform_to_dict(self, taglist, toklist, spans):
        '''
        Transfors a list of tokens and a list of tags into a dict{tag_type:[tuple(token1Text,token1Tag), ...]}
        :param taglist: The list of tags representing a single doc
        :param toklist: The list of tokens representing a single doc
        :return:
        '''
        assert(len(taglist) == len(toklist))
        assert(len(spans) == len(taglist))
        token_dicts=list()
        for i, tag in enumerate(taglist):
            tok = toklist[i]
            begin = spans[i][0]
            end = spans[i][1]
            d = dict()

            if re.match('^\s+$', tok) and tag != "O":
                tag = "O" # If the algo tagged whitespace as non "O", reset to "O"

            d['text'] = tok
            d['label'] = tag
            d['start'] = begin
            d['stop'] = end
            token_dicts.append(d)
        return token_dicts

    def remove_negated_concepts(self, tagged_docs):
        print "Finding negated concepts..."
        for docid, doc in tagged_docs.items():
            self.negexer.negate(doc)
        return tagged_docs

    def docs2json(self, tagged_documents):
        doc_dict = dict()
        for id, doc in tagged_documents.items():
            doc_dict[id] = dict({"NER_labels": doc.NER_token_labels, "text": doc.text})
        return json.dumps(doc_dict, ensure_ascii=False)