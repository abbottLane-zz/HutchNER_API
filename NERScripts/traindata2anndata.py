"""This script takes in a TSV/CSV file (EG one output by txt2tsv.py), and a tsv file containing the patient-level
gold standard from which to extract the diagnosis name, finds all documents with occurrences matching the diagnosis
description.

 Alternatively, this script could be replaced by a manual process over the first input.
 A long, arduous manual process....

input: tsv containing entire training set
input: tsv containing patient level gold info
return: a tsv containing only documents we assume to have dates of diagnosis. Cross your fingers.
"""

