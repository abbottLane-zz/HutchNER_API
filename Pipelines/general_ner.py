# Copyright (c) 2016-2017 Fred Hutchinson Cancer Research Center
#
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#
from flask import json

from DataLoading.JSONDataLoader import JSONDataLoader


def docs2json(docs):
    doc_dict=dict()
    for id, doc in docs.items():
        doc_dict[doc.document_id]=dict()
        doc_dict[doc.document_id]['text'] = doc.text
        doc_dict[doc.document_id]['NER_labels'] = list()
        for tok in doc.tokens:
            if tok.ent_type_ == "":
                entity = "O"
            else:
                entity = tok.ent_type_
            doc_dict[doc.document_id]['NER_labels'].append({
                'start':tok.idx,
                'stop':tok.idx + len(tok.orth_),
                'text':tok.orth_,
                'confidence':1,
                'label': entity
            })
    return json.dumps(doc_dict, ensure_ascii=False)


def main(documents, model):
    text_dl = JSONDataLoader(documents=documents)
    docs = text_dl.preprocess(spacy_model=model)
    json_response = docs2json(docs)
    return json_response

if __name__ == '__main__':
    # initialize large models on server startup
    import en_core_web_sm
    spacy_model = en_core_web_sm.load()
    documents = {"1":"HISTORY OF PRESENT ILLNESS: Mr. Bob is a 1000000-year-old gentleman with coronary artery disease, hypertension, hypercholesterolemia, COPD and tobacco abuse. He reports doing well. He did have some more knee pain for a few weeks, but this has resolved. He is having more trouble with his sinuses. I had started him on Flonase back in December. He says this has not really helped. Over the past couple weeks he has had significant congestion and thick discharge. No fevers or headaches but does have diffuse upper right-sided teeth pain. He denies any chest pains, nausea, PND, orthopnea, edema or syncope. His breathing is doing fine. No cough. He continues to smoke about half-a-pack per day. He plans on trying the patches again.\
\
CURRENT MEDICATIONS: Updated on CIS. They include aspirin, atenolol, Lipitor, Advair, Spiriva, albuterol and will add Singulair today.\
\
ALLERGIES: Sulfa caused a rash."}

    response =main(documents=documents, model=spacy_model)