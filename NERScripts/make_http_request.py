# Copyright (c) 2016-2017 Fred Hutchinson Cancer Research Center
#
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#
import os
import requests
from flask import json
from requests.auth import HTTPBasicAuth

# NOTE: the very last part of the url below dictates the model to be used for tagging.
#     Currently there are only three models to choose from:
#     "crf_ner": A problems, Treatments, Tests CRF model
#     "lstm_ner": A problems, Treatments, Tests LSTM model
#     "deid_crf": A deidentification CRF model
url = 'https://nlp-brat-prod01.fhcrc.org/hutchner/ner_neg/crf_ner'
data = {
    "1234": "the patient experienced no chest pressure or pain or dyspnea, or pain, or dyspnea"
}
headers = {"Content-Type: application/json"}
response = requests.get(url, json=data)

p_response = json.loads(response.text)
print json.dumps(p_response, sort_keys=True, indent=2)
