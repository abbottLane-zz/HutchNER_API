# Copyright (c) 2016-2017 Fred Hutchinson Cancer Research Center
#
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#
import os

import requests
from flask import json
from os.path import isfile, join
from requests.auth import HTTPBasicAuth


def load_data(data_dir):
    data=dict()
    onlyfiles = [f for f in os.listdir(data_dir) if isfile(join(data_dir, f))]
    for file in onlyfiles:
        with open(os.path.join(data_dir, file), "rb") as f:
            text = f.read()
            data[file]=text
    return data

url = 'https://nlp-brat-prod01.fhcrc.org/hutchner/ner_neg/crf'
data_dir = "/home/wlane/nethome/i2b2_data/2010_concepts_plusFH/test/txt"
# data = load_data(data_dir)
data = {
        "1234":"the patient experienced no chest pressure or pain or dyspnea, or pain, or dyspnea"
    }
headers = {"Content-Type: application/json"}
response = requests.get(url, json=data,auth=HTTPBasicAuth("wlane","python"))

p_response = json.loads(response.text)
print json.dumps(p_response, sort_keys = True, indent=2)

