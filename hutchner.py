#!flask/bin/python
from flask import Flask, make_response, jsonify, request, render_template, g

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
@app.route('/')
def index():
    return "Welcome to HutchNER! (T)"


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

if __name__ == '__main__':
    app.run()