# Copyright (c) 2016-2017 Fred Hutchinson Cancer Research Center
#
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#
from DataLoading.JSONDataLoader import JSONDataLoader
from NERExtraction.Extraction import NERExtraction


def main(documents, model_type, models):
    text_dl = JSONDataLoader(documents=documents)
    docs = text_dl.preprocess(spacy_model=models['spacy'])

    extractor = NERExtraction(docs, model_algo=model_type)
    tagged_documents = extractor.tag_all(models)
    json_response = extractor.docs2json(tagged_documents)
    return json_response

if __name__ == '__main__':
    main(documents=None, model_type=None, models=None)
