# Copyright (c) 2016-2017 Fred Hutchinson Cancer Research Center
#
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#
import os


class HTMLPrinter(object):

    def write_readable_prediction_results(self, list_of_documents, dir=os.getcwd(), model_algo="CRF"):
            html_docs = self.docs2html(list_of_documents)
            for id, doc in html_docs.items():
                with open(os.path.join(dir, model_algo+ "_"+id+".html"), "wb") as f:
                    f.write(doc)

    def docs2html(self, dict_of_documents):
        html_docs = dict()
        for id, d in dict_of_documents.items():
            html_docs[id]=d.doc2html()
        return html_docs