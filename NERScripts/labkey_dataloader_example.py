# Copyright (c) 2016-2017 Fred Hutchinson Cancer Research Center
#
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#
####### Load Labkey Data ####
from HutchNER.DataLoading.LabKeyDataLoading import LabKeyDataLoader
from HutchNER.NERPreprocessing.DocumentPreprocessing import UnformattedDocumentPreprocessor
from HutchNER.NERUtilities.MiscFunctions import load_labkey_server_info_from_ini

data_source_info = load_labkey_server_info_from_ini('labkey_sec_name')
lk_dl = LabKeyDataLoader(data_source_info["driver"], data_source_info["database"],
                                                    data_source_info["server"], data_source_info["table"],
                                                    data_source_info["job_run_ids"])
docs = lk_dl.load()
UnformattedDocumentPreprocessor(docs)
annotated_data = docs
detected_labels = lk_dl.get_detected_labels()