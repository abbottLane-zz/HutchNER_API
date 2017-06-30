import time

from HutchNER.NERPreprocessing.DocumentPreprocessing import i2b2DocumentPreprocessor

from HutchNER.DataLoading.i2b2DataLoading import i2b2DataLoader
from HutchNER.NEREvaluation.Evaluation import NEREvaluator
from HutchNER.NERExtraction.Extraction import NERExtraction
from HutchNER.NERUtilities import ArgumentParsingSettings
from HutchNER.NERUtilities.DocumentPrinter import HTMLPrinter


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
