# Copyright (c) 2013-2016 Fred Hutchinson Cancer Research Center
#
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#
import os
import jnius_config
from jnius import autoclass


def uw_sectioner(documents):
    #assert(documents)
    cwd = os.path.dirname(__file__)
    jnius_config.add_classpath(os.path.join("./Java/*"))
    jnius_config.add_classpath(os.path.join(cwd,os.path.join("Java","*")))
    out_file="/nethome/Clinical_Data/NER_Training_Corpus/sectioned",
    Chunker = autoclass('edu.uw.bhi.bionlp.sectionchunker.decode.App')
    chunker = Chunker()
    model_path_and_prefix = os.path.join(cwd, "Model","uw-bhi-bionlp-clinical-record-section-chunker")
    log_file = os.path.join(cwd, "Log", "log.sectionchunker.out")
    config_file = os.path.join(cwd, "Config", "sample_decode_config.xml")
    args = ["--model-path-and-prefix", model_path_and_prefix,
            "--input-format", "TEXT",
            "--input-directory", "/home/wlane/nethome/Clinical_Data/NER_Training_Corpus/txt",
            "--output-directory", "/nethome/Clinical_Data/NER_Training_Corpus/sectioned",
            "--log-file", log_file,
            "--logging",
            "--config-file", config_file
            ]
    chunker.main(args)
    return out_file


if __name__ == "__main__":
    uw_sectioner(documents=None)
