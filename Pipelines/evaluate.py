# Copyright (c) 2016-2017 Fred Hutchinson Cancer Research Center
#
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#

import time
from DataLoading.i2b2DataLoading import i2b2DataLoader
from NEREvaluation.Evaluation import NEREvaluator
from NERExtraction.Extraction import NERExtraction
from NERUtilities import ArgumentParsingSettings

from NERUtilities.DocumentPrinter import HTMLPrinter


def main():
    """ Entry point to HutchNER1: Concept NERExtraction Training """
    # start timer
    start = time.clock()

    # Parse incoming cmd line arguments
    args = ArgumentParsingSettings.get_testing_args()
    data_dir = args.datadir
    model_dir = args.model_dir
    local_annotations = args.annots
    labkey_ini_section = args.section

    # Load the documents
    text_dl = i2b2DataLoader(txt_dir = data_dir, annotation_dir=local_annotations)
    docs = text_dl.load()

    # Run NER driver with models and data provided in dirs
    extractor = NERExtraction(docs)
    tagged_documents = extractor.tag_all()
    neg_documents = extractor.remove_negated_concepts(tagged_documents)


    # Create DocumentPrinter object; print/write document objects in desired format
    dp = HTMLPrinter()
    dp.write_readable_prediction_results(neg_documents,"/home/wlane/PycharmProjects/HutchNER1/HutchNER1/NERResults")

    # Evaluate the performance on TAGGED DOCUMENTS (not the negated ones)
    labels = extractor.possible_labels
    ev = NEREvaluator(tagged_documents, labels)
    ev.write_results("/home/wlane/PycharmProjects/HutchNER1/HutchNER1/NEREvaluation/EvalResults", strictness="exact")
    ev.write_results("/home/wlane/PycharmProjects/HutchNER1/HutchNER1/NEREvaluation/EvalResults", strictness="overlap")

    # Print time elapsed to console
    end = time.clock()
    print "##################################"
    print " \tTime Elapsed: " + str(int((end-start)/60)) + " minutes and " + str(int((end-start) % 60)) + " seconds."
    print "##################################"

if __name__ == '__main__':
    main()
