"""
RPT to CSV:
This script takes 2 arguments:
                1. -f, file path to text file that you want to process
                2. -d, the delimiter string that separates the documents in the rpt file
                3. -o, the output directory where to write the tsv/csv file
And produces a CSV file of all documents

-wlane
"""
import codecs
import argparse
import csv


def main():
    parser = argparse.ArgumentParser(description='This program converts Emily\'s .rpt SQL dumps to CSV, given the '
                                     'directory of the file and the delimiter string used')
    parser.add_argument('-f', '--file', help='The full path to the file you want to process', required=True)
    parser.add_argument('-d', '--delimiter', help='The string that serves as a delimiter between documents', required=True)
    parser.add_argument('-o', '--outfile', help='The directory to which you wish to write the output csv file', required=True)
    args = vars(parser.parse_args())


    file = args['file']
    delim = args['delimiter']
    out = args['outfile']

    docs = load_docs(file, delim)
    write_docs(out, docs)
        

def load_docs(file, delim):
    with codecs.open(file, "rb", encoding="utf-16") as f:
        alldocs = f.read()

    document_list = alldocs.split(delim)
    count = 0
    sectioned_doc_list = list()
    for doc in document_list:
        count += 1
        sections = doc.lstrip().split("\t")
        sections = filter(None, sections)
        sectioned_doc_list.append(sections)
    return sectioned_doc_list


def write_docs(out, docs):
    with open(out+"dx_docs_nofilter.tsv", "wb") as f:
        writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        for sections in docs:
            if len(sections) > 8:
                pid = sections[len(sections)-1]
                event_id = sections[1]
                doc_id = pid + "_" + event_id
                timestamp = sections[2]
                text = concat_text_sections(sections[8:-2])
                writer.writerow([pid, doc_id, timestamp, text])


def concat_text_sections(section_list):
    full_text=""
    for sec in section_list:
        full_text += sec.encode("utf-8")
    return full_text

if '__main__' == __name__:
    main()