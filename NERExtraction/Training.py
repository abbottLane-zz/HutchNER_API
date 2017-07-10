# Copyright (c) 2016-2017 Fred Hutchinson Cancer Research Center
#
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#
import os

import scipy
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from NERUtilities.MiscFunctions import CLUSTER_PATH

from FeatureProcessing import sent2features
from NERUtilities.Clusters import Clusters
from sklearn.externals import joblib


class NERTrainer(object):
    """
    The Object driving the clinical concept extraction Training pipeline.
    """
    def __init__(self,docs, detected_labels, optimize_hyperparams=False):
        self.detected_labels = detected_labels
        self.annotated_data =docs
        self.clusters = Clusters(CLUSTER_PATH)
        self.model_path = os.path.join("..","NERResources", "Models")
        self.optimize_hyperparams = optimize_hyperparams

    def train(self):
        training_docs = list()
        print "Training Concept Extractor ... "
        for i, doc_id in enumerate(self.annotated_data):
            doc_obj = self.annotated_data[doc_id]
            training_docs.append(doc_obj)
        print "Possible labels: ", str(self.detected_labels)
        #for type in self.detected_labels:
        model_n = "_".join(self.detected_labels)
        model_name = os.path.join(self.model_path, "model-" + model_n + ".pk1")
        training_labels = self._get_training_labels(training_docs, self.detected_labels)
        x_train_list, y_train_list = self._get_features_and_labels(training_docs, training_labels)

        if self.optimize_hyperparams:
            self._tune_hyperparams(x_train_list, y_train_list, self.detected_labels)
        else:
            self._run_training(x_train_list, y_train_list, model_name)
            print("CRF model written to: " + model_name)

        return model_name

    def _tune_hyperparams(self, x_train, y_train, labels):
        print "Running hyperparam grid search (not training). This process fits the models many times, and can take a " \
              "long time. "
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            max_iterations=100,
            all_possible_transitions=True
        )
        params_space = {
            'c1': scipy.stats.expon(scale=0.5),
            'c2': scipy.stats.expon(scale=0.05),
        }
        # use the same metric for evaluation
        f1_scorer = make_scorer(metrics.flat_f1_score,
                                average='weighted', labels=labels)

        # search
        rs = RandomizedSearchCV(crf, params_space,
                                                    cv=3,
                                                    verbose=1,
                                                    n_jobs=-1,
                                                    n_iter=25,
                                                    scoring=f1_scorer)
        rs.fit(x_train, y_train)
        print('best params:', rs.best_params_)
        print('best CV score:', rs.best_score_)
        print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

    def _run_training(self, x_train, y_train, model_name):
        # for xseq, yseq in zip(x_train, y_train):
        #     trainer.append(xseq, yseq)
        print "Setting CRF Params for training " + model_name
        crf = sklearn_crfsuite.CRF(
            c1= 1.0,  # coefficient for L1 penalty
            c2= 1e-3,  # coefficient for L2 penalty
            max_iterations= 200,  # stop earlier
            # include transitions that are possible, but not observed
            all_possible_transitions= True
        )
        print "Training " + model_name
        crf.fit(x_train,y_train)  # produces model file with this name
        joblib.dump(crf, model_name)


    def _get_features_and_labels(self, docs, labels):
        print "Fetching feature vectors ..."
        # Grab all Spacy preprocessed features
        tagged_sents = list()
        for doc in docs:
            try:
                tokens = doc.tokens
                tagged_sents.append(tokens)
            except Exception as e:
                print str(e)

        x = [sent2features(s, self.clusters) for s in tagged_sents]
        y = labels  # labels in doesn't change
        return x, y

    def _get_training_labels(self, training_docs, labels):
        print "Fetching training labels for annotation type: " + str(labels)
        all_label_vecs = list()
        count = 0
        for doc in training_docs:
            all_label_vecs.append(doc.get_crf_training_vectors(labels))
            count +=1
        return all_label_vecs

    def _get_model_type_name(self, detected_labels):
        string =""
        for label in detected_labels:
            string += str(label)+"_"
        return string



