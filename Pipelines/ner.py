from DataLoading.JSONDataLoader import JSONDataLoader
from NERExtraction.Extraction import NERExtraction


def main(documents, model_type, spacy_model):
    text_dl = JSONDataLoader(documents=documents)
    docs = text_dl.preprocess(spacy_model=spacy_model)

    extractor = NERExtraction(docs, model_algo=model_type)
    tagged_documents = extractor.tag_all()
    json_response = extractor.docs2json(tagged_documents)
    return json_response

if __name__ == '__main__':
    main(documents=None, model_type=None, spacy_model=None)
