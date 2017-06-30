import time

from HutchNER.DataLoading.i2b2DataLoading import i2b2DataLoader

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
    detected_labels = i2b2_dl.get_detected_labels()

    trainer = NERTrainer(docs, detected_labels)
    model_dir = trainer.train()

    end = time.clock()
    print "##################################"
    print " Training summary:\n\t 1 model trained"
    print " \tTime Elapsed: " + str(int((end-start)/60))+ " minutes and " + str(int((end-start)%60)) + " seconds."
    print "\tModel written to " + model_dir
    print "##################################"

if __name__ == '__main__':
    main()
