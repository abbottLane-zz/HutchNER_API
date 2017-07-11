# Copyright (c) 2016-2017 Fred Hutchinson Cancer Research Center
#
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#
import os

import requests
from flask import json
from os.path import isfile, join
from requests.auth import HTTPBasicAuth


url = 'https://nlp-brat-prod01.fhcrc.org/hutchner/ner_neg/crf'
data = {
        "1234":"the patient experienced no chest pressure or pain or dyspnea, or pain, or dyspnea"
    }
headers = {"Content-Type: application/json"}
response = requests.get(url, json=data,auth=HTTPBasicAuth("wlane","python"))

p_response = json.loads(response.text)
print json.dumps(p_response, sort_keys = True, indent=2)

