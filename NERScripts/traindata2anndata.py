# Copyright (c) 2016-2017 Fred Hutchinson Cancer Research Center
#
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#
"""This script takes in a TSV/CSV file (EG one output by txt2tsv.py), and a tsv file containing the patient-level
gold standard from which to extract the diagnosis name, finds all documents with occurrences matching the diagnosis
description.

 Alternatively, this script could be replaced by a manual process over the first input.
 A long, arduous manual process....

input: tsv containing entire training set
input: tsv containing patient level gold info
return: a tsv containing only documents we assume to have dates of diagnosis. Cross your fingers.
"""

