import time

from HutchNER.DataLoading.LabKeyDataLoading import LabKeyDataLoader
from HutchNER.NERPreprocessing.DocumentPreprocessing import UnformattedDocumentPreprocessor
from HutchNER.NERUtilities.MiscFunctions import load_labkey_server_info_from_ini
from HutchNER.NERExtraction.Extraction import NERExtraction
from HutchNER.NERUtilities import ArgumentParsingSettings
from HutchNER.NERUtilities.DocumentPrinter import DocumentPrinter


def main():
    # start timer
    start = time.clock()

    args = ArgumentParsingSettings.get_labkey_predict_args()
    section = args.section
    model_type = args.model_type

    data_source_info = load_labkey_server_info_from_ini(section)
    lk_dl = LabKeyDataLoader(data_source_info["driver"], data_source_info["database"],
                                 data_source_info["server"], data_source_info["table"],
                                 data_source_info["job_run_ids"])
    docs = lk_dl.load()
    annotations_only = lk_dl.get_annotations()

    tester = NERExtraction(docs, model_algo=model_type)
    tagged_documents = tester.tag_all()

    dp = DocumentPrinter(tagged_documents, algo_type=model_type)
    dp.write_readable_prediction_results()


    end = time.clock()
    print "##################################"
    print " \tTime Elapsed: " + str(int((end-start)/60))+ " minutes and " + str(int((end-start)%60)) + " seconds."
    print "##################################"

if __name__ == '__main__':
    main()
