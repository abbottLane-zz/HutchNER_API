#!/usr/bin/env python

# Copyright (c) 2016-2017 Fred Hutchinson Cancer Research Center
#
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#
import os
import time
import optparse
import numpy as np
from loader import prepare_sentence
from utils import create_input, iobes_iob, zero_digits

def main(document_objs, model):
    pred_tuples_by_doc_id = dict()
    start = time.time()
    print 'Tagging...'

    doc_count = 0

    for id, doc in document_objs.items():
        doc_count +=1
        pred_tuples_by_doc_id[id]=tag_document(doc, model['parameters'], model['model'], model['f_eval'], model['word_to_id'], model['char_to_id'])

    print '---- %i documents tagged in %.4fs ----' % (doc_count, time.time() - start)
    return pred_tuples_by_doc_id


def tag_document(doc, parameters, model, f_eval, word_to_id, char_to_id):
    count = 0
    all_ypreds = list()
    all_tokens = list()
    for line in doc.sentences:
        toks_text = [x.orth_ for x in line.tokens]
        # line = ' '.join(toks_text)
        if toks_text:  # WL edit: used to be 'if line', was crashing on '\n' lines
            # Lowercase sentence
            if parameters['lower']:
                toks_text = [line.lower() for line in toks_text]
            # Replace all digits with zeros
            if parameters['zeros']:
                toks_text = [zero_digits(line) for line in toks_text]
            # Prepare input
            sentence = prepare_sentence(toks_text, word_to_id, char_to_id,
                                            lower=parameters['lower'])
            input = create_input(sentence, parameters, False)
            # Decoding
            if parameters['crf']:
                y_preds = np.array(f_eval(*input))[1:-1]
            else:
                y_preds = f_eval(*input).argmax(axis=1)
            y_preds = [model.id_to_tag[y_pred] for y_pred in y_preds]
            # Output tags in the IOB2 format
            if parameters['tag_scheme'] == 'iobes':
                y_preds = iobes_iob(y_preds)
            # Write tags
            assert len(y_preds) == len(toks_text)

            all_ypreds.append(y_preds)
            all_tokens.append(toks_text)

        count += 1
        if count % 100 == 0:
            print count

    return (all_ypreds,all_tokens)

if __name__=="__main__":
    optparser = optparse.OptionParser()
    optparser.add_option(
        "-m", "--model", default="",
        help="Model location"
    )
    optparser.add_option(
        "-i", "--input", default="",
        help="Input file location"
    )
    optparser.add_option(
        "-o", "--output", default="./",
        help="Output file location"
    )
    optparser.add_option(
        "-d", "--delimiter", default="__",
        help="Delimiter to separate words from their tags"
    )
    opts = optparser.parse_args()[0]
    # Check parameters validity
    assert opts.delimiter
    assert os.path.isdir(opts.model)
    # assert os.path.isfile(opts.input) ## no longer a requirement: can accept dir or file
    main(data_dir=opts.input, model_dir=opts.model)