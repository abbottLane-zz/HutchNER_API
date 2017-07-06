# Copyright (c) 2016-2017 Fred Hutchinson Cancer Research Center
#
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#
import time

from HutchNER.DataLoading.i2b2DataLoading import i2b2DataLoader

from HutchNER.NERPreprocessing.DocumentPreprocessing import i2b2DocumentPreprocessor

from HutchNER.NERExtraction.Training import NERTrainer
from HutchNER.NERUtilities import ArgumentParsingSettings


def main():
    """ Entry point to GP Concept NERExtraction Training System """
    # start timer
    start = time.clock()

    # Parse incoming cmd line arguments
    args = ArgumentParsingSettings.get_training_args()
    data_dir = args.datadir
    local_annotations = args.annots

    #load and preprocess the data
    i2b2_dl = i2b2DataLoader(data_dir, local_annotations)
    docs = i2b2_dl.load()
    i2b2DocumentPreprocessor(docs)
    i2b2_dl.join_annotations(docs)
    detected_labels = i2b2_dl.get_detected_labels()

    trainer = NERTrainer(docs, detected_labels, optimize_hyperparams=True)
    model_dir_by_tag_type = trainer.train()

    end = time.clock()
    print "##################################"
    print " Training summary:\n\t" + str(len(model_dir_by_tag_type)) + " models trained"
    print " \tTime Elapsed: " + str(int((end-start)/60))+ " minutes and " + str(int((end-start)%60)) + " seconds."
    for tag_type in model_dir_by_tag_type:
        print "\tModel \'"+tag_type+ "\' written to " + model_dir_by_tag_type[tag_type]
    print "##################################"

if __name__ == '__main__':
    main()
