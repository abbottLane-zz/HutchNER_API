#!flask/bin/python
import os

import requests
from flask import Flask, make_response, jsonify, request, render_template, g, json
from os.path import isfile, join

from Pipelines import ner_negation, ner
from flask_oauthlib.provider import OAuth2Provider
import en_core_web_sm

app = Flask(__name__)
oauth=OAuth2Provider(app)

# initialize spacy for preprocessing
spacy_model = en_core_web_sm.load()

#################
### Endpoints ###
#################
@app.route('/ner/<string:alg_type>', methods=['GET'])
def ner_pipeline(alg_type):
    documents = request.json
    if documents:
        json_response = ner.main(documents, alg_type, spacy_model)
        return str(json_response)
    else:
        return make_response(jsonify({'error': 'No data provided'}), 400)


@app.route('/ner_neg/<string:alg_type>', methods=['GET'])
def ner_negation_pipeline(alg_type):
    documents = request.json
    if documents:
        json_response = ner_negation.main(documents, alg_type, spacy_model)
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
@app.route('/demo/')
def index():
    return render_template('index.html', **locals())


@app.route('/demo/', methods=['POST'])
def submit_textarea():
    url = 'https://nlp-brat-prod01.fhcrc.org/hutchner/ner_neg/crf'
    data={"1":request.data}
    headers = {"ontent-Type": "application/json"}
    response = requests.get(url, json=data, headers=headers)
    p_response = json.loads(response.text)
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
    colors = {"problem":"#ff6174","treatment":"#9df033", "test":"#61e9ff"}
    header=""
    tokens = json['1']['NER_labels']
    text = json['1']['text']
    in_span=False
    for token in tokens:
        if not in_span:
            if token['label'] != "O":
                in_span=True
                header+= "<span type=\""+token['label']+"\" style=\"background-color:"+colors[token['label']]+ insert_negation(token)+"\">"

        if in_span:
            if token['label'] =="O":
                in_span = False
                header+="</span>"
        header += token['text'] + " "

        if token['text'] == ".":
            header += "<br>"
    return header


def insert_negation(token):
    if "negation" in token:
        return ";color:#f44141 "
    else:
        return ""


if __name__ == '__main__':
    app.run()