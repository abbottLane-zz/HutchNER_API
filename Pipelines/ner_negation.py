import os

from flask import json
from os.path import isfile, join

from DataLoading.JSONDataLoader import JSONDataLoader
from NERExtraction.Extraction import NERExtraction


def main(documents, model_type, spacy_model):
    text_dl = JSONDataLoader(documents=documents)
    docs = text_dl.preprocess(spacy_model=spacy_model)

    extractor = NERExtraction(docs, model_algo=model_type)
    tagged_documents = extractor.tag_all()
    tagged_documents = extractor.remove_negated_concepts(tagged_documents)
    json_response = extractor.docs2json(tagged_documents)
    print "Done."
    return json_response

if __name__ == '__main__':

    def load_data(data_dir):
        data = dict()
        onlyfiles = [f for f in os.listdir(data_dir) if isfile(join(data_dir, f))]
        for file in ["0467.txt"]:
            with open(os.path.join(data_dir, file), "rb") as f:
                text = f.read()
                data[file] = text.decode('utf-8')
        return data

    import en_core_web_sm
    spacy_model = en_core_web_sm.load()
    data_dir = "/home/wlane/nethome/i2b2_data/2010_concepts_plusFH/test/txt"
    #docs= load_data(data_dir)
    docs = {
         "1234":"the patient experienced no chest pressure or pain or dyspnea, or pain, or dyspnea, or pain"
    }
    response = main(documents=docs, model_type='crf', spacy_model=spacy_model)
    print json.dumps(response, sort_keys=True, indent=2)

