#!flask/bin/python
''' author@wlane '''
# Copyright (c) 2016-2017 Fred Hutchinson Cancer Research Center
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os

import requests
from flask import Flask, make_response, jsonify, request, render_template, g, json
from os.path import isfile, join

from sklearn.externals import joblib

from LSTMExec.model import Model
from Pipelines import ner_negation, ner
from flask_oauthlib.provider import OAuth2Provider
import en_core_web_sm

def load_lstm_model(model_dir):
    model = Model(model_path=model_dir)
    # Load existing model
    print "Loading model..."
    parameters = model.parameters

    # Load reverse mappings
    word_to_id, char_to_id, tag_to_id = [
        {v: k for k, v in x.items()}
        for x in [model.id_to_word, model.id_to_char, model.id_to_tag]
    ]

    # Load the model
    _, f_eval = model.build(training=False, **parameters)
    model.reload()
    return {"model":model,
            "f_eval":f_eval,
            "word_to_id":word_to_id,
            "char_to_id":char_to_id,
            "tag_to_id":tag_to_id,
            "parameters":parameters}



# initialize large models on server startup
spacy_model = en_core_web_sm.load()
lstm_ner_model= load_lstm_model(model_dir=os.path.join(os.path.dirname(__file__), os.path.join("LSTMExec","models","i2b2_fh_50_newlines")))
crf_ner_model= joblib.load(os.path.join(os.path.dirname(__file__), os.path.join("NERResources","Models", "model-test_problem_treatment_.pk1")))
models={"crf_ner":crf_ner_model, "lstm_ner":lstm_ner_model, "spacy":spacy_model}



app = Flask(__name__)
oauth=OAuth2Provider(app)




#################
### Endpoints ###
#################
@app.route('/ner/<string:alg_type>', methods=['GET'])
def ner_pipeline(alg_type):
    documents = request.json
    if documents:
        json_response = ner.main(documents, alg_type, models)
        return json_response.encode('utf-8')
    else:
        return make_response(jsonify({'error': 'No data provided'}), 400)


@app.route('/ner_neg/<string:alg_type>', methods=['GET'])
def ner_negation_pipeline(alg_type):
    documents = request.json
    if documents:
        json_response = ner_negation.main(documents, alg_type, models)
        return json_response.encode('utf-8')
    else:
        return make_response(jsonify({'error': 'No data provided'}), 400)


@app.route('/section_detection', methods = ['GET'])
def section_detection_pipeline():
    return jsonify({"NotImplementedError": "Section detection endpoint is not yet hooked up. Sorry!"})


@app.errorhandler
def not_found():
    return make_response(jsonify({'error': 'Not found'}), 404)


#########################
### HutchNER Demo App ###
#########################

# @app.route('/demo/', methods=['GET'])
# def handle_data():
#     projectpath = request.form['testval']
#     print "HEY! i got it:" + str(projectpath)

@app.route('/demo/')
def index():
    return render_template('index.html', **locals())


@app.route('/demo/', methods=['POST', 'GET'])
def submit_textarea():
    url = 'https://nlp-brat-prod01.fhcrc.org/hutchner/ner_neg/'
    json_request_data = json.loads(request.data)
    algo_type = json_request_data['algo']
    url += algo_type
    data={"1":json_request_data['text']}
    headers = {"content-type": "application/json"}
    response = requests.get(url, json=data, headers=headers)
    p_response = json.loads(response.text)
    #contentType: 'text/plain',
    return json2html(p_response)


def load_data(data_dir):
    data=dict()
    onlyfiles = [f for f in os.listdir(data_dir) if isfile(join(data_dir, f))]
    for file in onlyfiles:
        with open(os.path.join(data_dir, file), "rb") as f:
            text = f.read()
            data[file]=text
    return data


def json2html(json):
    problem_color = "#DDA0DD"
    treatment_color = "#9df033"
    test_color = "#61e9ff"
    colors = {"problem": problem_color,
              "treatment": treatment_color,
              "test": test_color,
              "b-problem": problem_color,
              "i-problem": problem_color,
              "b-treatment": treatment_color,
              "i-treatment": treatment_color,
              "b-test": test_color,
              "i-test": test_color}
    header="<span style=\"color:#f44141\">Definite Negated</span> " \
           "<span style=\"color:#ff7c00\">Probable Negated</span> " \
           "<span style=\"color:#ffec48\">Ambivalent Negated</span> " \
           "<span style=\"background-color:#DDA0DD\">Problem</span> " \
           "<span style=\"background-color:#9df033\">Treatment</span> " \
           "<span style=\"background-color:#61e9ff\">Test</span><br><br> "
    tokens = json['1']['NER_labels']
    in_span=False
    for token in tokens:
        if not in_span:
            if token['label'] != "O":
                in_span=True
                header+= "<span type=\""+token['label']+"\" style=\"background-color:"+colors[token['label'].lower()]+ insert_negation_color(token)+"\">"

        if in_span:
            if token['label'] =="O":
                in_span = False
                header+="</span>"
        header += token['text'] + " "

        if token['text'] == ".":
            header += "<br>"
    return header


def insert_negation_color(token):
    if "negation" in token:
        if "DEFINITE" in token["negation"]:
            return ";color:#f44141 "
        elif "AMBIVALENT" in token["negation"]:
            return ";color:#ffec48"
        else:
            return ";color:#ff7c00"
    else:
        return ""


if __name__ == '__main__':
    app.run()
