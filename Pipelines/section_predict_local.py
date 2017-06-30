import time

from DataLoading.SectionerXMLDataLoading import SectionerXMLDataLoader
from Sectioning.ClinicalSectioner import uw_sectioner

from NERExtraction.Extraction import NERExtraction
from NERUtilities import ArgumentParsingSettings
from NERUtilities.DocumentPrinter import HTMLPrinter


def main():
    # start timer
    start = time.clock()

    # Parse incoming cmd line arguments
    args = ArgumentParsingSettings.get_local_predict_args()
    data_dir = args.datadir
    model_type = args.model_type

    # Section raw documents
    sectioner_out_dir = uw_sectioner(data_dir)

    # Load sectioned docs
    xml_dl = SectionerXMLDataLoader(xml_dir=sectioner_out_dir, clean_tmp_files=True)
    docs = xml_dl.load()

    # Perform NER on sectioned docs
    extractor = NERExtraction(docs, model_algo=model_type)
    tagged_documents = extractor.tag_all()
    tagged_documents = extractor.remove_negated_concepts(tagged_documents)

    # Print full docs
    dp = HTMLPrinter()
    dp.write_readable_prediction_results(tagged_documents, "/home/wlane/PycharmProjects/HutchNER/HutchNER/NERResults", model_algo=model_type)

    end = time.clock()
    print "##################################"
    print " \tTime Elapsed: " + str(int((end-start)/60))+ " minutes and " + str(int((end-start)%60)) + " seconds."
    print "##################################"

if __name__ == '__main__':
    main()