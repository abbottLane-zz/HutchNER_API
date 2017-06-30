import csv
import os

import spacy
import datetime
from fuzzyparsers import parse_date, re


def main():
    print "Loading spacy..."
    nlp_tk = spacy.load('en')
    print "Loaded spacy."
    gold = get_gold_data()
    print("Loading and NER-tagging documents ... ")
    docs_by_patient = get_docs_by_patient()
    print("Searching docs for datetime mentions...this takes a while...")
    timediff_docs_by_patient = filter_by_timediff(gold, docs_by_patient, nlp_tk)
    print("Writing filtered dx_doc tsv file to ../NERResources/")
    write_docs('../NERResources/Data/', timediff_docs_by_patient)


def get_gold_data():
    gold = list()
    with open('../NERResources/RawData/300PtsWithDxInfo.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            gold.append(row)
    return gold


def get_docs_by_patient():
    docs_by_patient = dict()
    with open('../NERResources/RawData/dx_docs_nofilter.tsv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            if row[0] not in docs_by_patient:
                docs_by_patient[row[0]] = list()
            docs_by_patient[row[0]].append(row)
    return docs_by_patient


def patient_date_pair_examined(patient_date_pairs, pid, d1):
    if pid in patient_date_pairs:
        if d1 in patient_date_pairs[pid]:
            return True
        else:
            patient_date_pairs[pid][d1] = True
    else:
        patient_date_pairs[pid] = dict()
        patient_date_pairs[pid][d1] = True
    return False


def filter_by_timediff(gold, docs_by_patient, nlp_tk):
    filtered_docs_by_patient = dict()
    patient_date_pairs = {}
    for patient in gold:
        if len(patient) > 6:
            pid = patient[1]
            d1 = patient[5]
            status = patient[7]
            if status != "No Change": # This is a status we dont care to try and detect
                if pid in docs_by_patient and not patient_date_pair_examined(patient_date_pairs, pid, d1):
                    list_of_docs = docs_by_patient[pid]
                    # foreach doc, go through the text, find reference to this date, or approximate close dates
                    for doc in list_of_docs:
                        text = doc[3]
                        dates = get_dates_from_ner(nlp_tk, text)
                        regex_dates = get_regex_dates(text)
                        # Compile all dates into single list
                        all_dates = dates.union(regex_dates)

                        # convert dates to datetime objs
                        dto_gold = parse_date(d1)
                        dtos_to_match = string2date(all_dates)

                        # Check for fuzzy match
                        has_match = has_fuzzy_date_match(dto_gold, dtos_to_match)
                        if has_match:
                            if doc[0] not in filtered_docs_by_patient: # if patient id does not exist...
                                filtered_docs_by_patient[doc[0]] = dict()
                                filtered_docs_by_patient[doc[0]][doc[1]] = doc
                            elif doc[1] not in filtered_docs_by_patient[doc[0]]: # make sure docid doesnt already exist...
                                filtered_docs_by_patient[doc[0]][doc[1]] = doc
    return filtered_docs_by_patient


def write_docs(out_dir, docs_by_patient):
    alldocs = list()
    tmp_list = list()
    count = 0
    for patient, documents in docs_by_patient.items():
        for docid, doc in documents.items():
            if count % 100 == 0:
                alldocs.append(tmp_list)
                tmp_list = list()
            tmp_list.append(doc)
            count += 1
    alldocs.append(tmp_list)
    write(alldocs, out_dir)


def write(alldocs, out_dir):
    for i, docs in enumerate(alldocs):
        with open(os.path.join(out_dir, "dx_status_docs_"+str(i)+".nlp.csv"), "wb") as f:
            writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["MRN", "FillerOrderNo", "EventDate", "ObservationValue"])
            for d in docs:
                pid = d[0]
                doc_id = d[1]
                timestamp = d[2]
                text = d[3]
                writer.writerow([pid, doc_id, timestamp, text])


def has_fuzzy_date_match(gold_date, dates2match):
    for date in dates2match:
        timediff = abs(gold_date-date)
        if timediff.days < 2:
            return True
    return False


def get_regex_dates(text):
    regex_dates = set()
    month_date_pattern = re.compile("(([Jj]anuary|[Ff]ebruary|[Mm]arch|[Aa]pril|[Mm]ay|[Jj]une|[Jj]uly|[Aa]ugust|[Ss]eptember|[Oo]ctober|[Nn]ovember|[dD]ecember)[, ]*\d*[, ]*[\d]+)")
    number_patterns = re.compile("(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})")
    month_only = re.compile("([Jj]anuary|[Ff]ebruary|[Mm]arch|[Aa]pril|[Mm]ay|[Jj]une|[Jj]uly|[Aa]ugust|[Ss]eptember|[Oo]ctober|[Nn]ovember|[dD]ecember)")

    res1 = month_date_pattern.findall(text)
    res2 = number_patterns.findall(text)
    res3 = month_only.findall(text)

    for res in res1:
        regex_dates.add(res[0])
    for res in res2:
        regex_dates.add(res)
    for res in res3:
        regex_dates.add(res)

    return regex_dates


def get_dates_from_ner(nlp_tk, text):
    dates=set()
    pr_text = nlp_tk(unicode(text.decode("utf-8")))
    for ent in pr_text.ents:
        if ent.label_ == "DATE":
            dates.add(ent.text)
    return dates


def string2date(date_set):
    dtos = list()
    for date in date_set:
        try:
            dto = parse_date(date)
            dtos.append(dto)
        except:
            # print "Could not parse \'" + date + "\' as a date."
            tmp=0
    return dtos

if '__main__' == __name__:
    main()




