# Copyright (c) 2016-2017 Fred Hutchinson Cancer Research Center
#
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#
import pyodbc
import warnings
from AbstractClasses import AbstractDataLoader
from DataClasses import Document, GoldAnnotation
from HutchNER.NERPreprocessing.DocumentPreprocessing import UnformattedDocumentPreprocessor


class LabKeyDataLoader(AbstractDataLoader):
    def __init__(self, driver, database, server, table, job_run_ids, do_get_annotations=False):
        super(LabKeyDataLoader, self).__init__()
        self.driver = driver
        self.database = database
        self.server = server
        self.table = table
        self.job_run_ids = job_run_ids
        self.do_get_annotations = do_get_annotations
        self.detected_annotations = set()

    def get_annotations(self):
        '''
        Public-pacing annotation getter. This should be used as info only, not for processing, as annotation formats
        differ depending on the source so we cannot garentee any standard format at tis point.

        The document objects retrieved by get_docs() will contain standardized annotations for downstream
        processing.
        :return: A list of dictionaries of {attrib_type<string>:value<string|int>}
        '''
        if self.annotations is not None:
            return self.annotations
        else:
            warnings.warn("There were no annotations retrieved in dataloading, so 'get_annotations()' returns nothing")
            return []

    def load(self):
        '''
        Using attributes from self, loads documents and annotations from LabKey into memory
        :return: True if load() succeeded, False otherwise
        '''
        docs = self.load_documents()
        UnformattedDocumentPreprocessor(docs)

        # Add annotations to document objects
        if self.do_get_annotations:
            self.join_annotations(docs)

        return docs

    def load_documents(self):
        '''
        Loads just the documents from LabKey server
        :return: list of Document objects
        '''
        doc_results = self.__get_labkey_data(self.job_run_ids,
                                                    self.table,
                                                    self.driver,
                                                    self.database,
                                                    self.server,
                                                    data_type="documents")
        doc_dict=dict()
        for docid, text in doc_results.items():
            doc_dict[docid] = Document(docid, text)

        return doc_dict

    def load_annotations(self):
        '''
        Loads just the annotations from Labkey server
        :return: List of GoldAnnotation objects
        '''
        ann_results= self.__get_labkey_data(self.job_run_ids,
                                                    self.table,
                                                    self.driver,
                                                    self.database,
                                                    self.server,
                                                    data_type="annotations")

        return ann_results

    def join_annotations(self, docs):
        if self.do_get_annotations:
            self.annotations = self.load_annotations()

            # Combine annotations to docs here.
            for ann_dict in self.annotations:
                document = docs[ann_dict['ReportNo']]
                self._add_annotation(document, ann_dict)

    def __get_labkey_data(self, job_ids, table, driver, database, server, data_type="documents"):
        '''
        :param job_ids: The list of job_id's you want to query from
        :param table: The table in the database that you want to query from
        :param driver: Your machine's installed ODBC driver name
        :param database: The name of the database you want to query from
        :param server: the address of the server where the database lives
        :param data_type: Either the "annotations" or the "documents"
        :return: list of Concept objects
        '''
        results = list()
        cur = self.__get_server_connection(driver, database, server)
        if data_type == "annotations":
            results = self.__execute_annotation_pull(cur, job_ids, table)  # table='SocialHistories'
        elif data_type == "documents":
            results = self.__execute_text_pull(cur, job_ids)
        return results

    def __get_server_connection(self, odbc_driver, db, server):
        connStr = (
            'DRIVER={' + odbc_driver + '};\
                SERVER=' + server + ';\
                DATABASE=' + db + ';\
                Trusted_Connection=yes')
        conn = pyodbc.connect(connStr)
        return conn.cursor()

    def __execute_annotation_pull(self, cur, job_ids, table):
        '''
        :param connection_curser: an active connection to an SQL database
        :param data_division: a list of jabkey job_run_ids to extract from labkey
        :param table: the table in the database that houses the data
        :return:
        '''
        query_string = "\
        SELECT nlp.FieldResult.*, nlp.Report.JobRunId, nlp.Report.MRN,\
            nlp.Report.ReportNo, nlp.Report.Status,\
            nlp.StartStopPosition.StartPosition, nlp.StartStopPosition.StopPosition \
        FROM nlp.FieldResult\
        JOIN nlp.StartStopPosition\
        ON nlp.FieldResult.FieldResultId = nlp.StartStopPosition.FieldResultId\
        JOIN nlp.Report \
        ON nlp.FieldResult.ReportId = nlp.Report.ReportId" + \
                       self.__AND_nlp_Report_JobRunID_IN(job_ids) + \
                       "AND nlp.Report.Status = 'approved'\
        AND nlp.FieldResult.TargetTable = '" + table + "'"
        cur.execute(query_string)
        columns = [column[0] for column in cur.description]
        results = []
        for row in cur.fetchall():
            results.append(dict(zip(columns, row)))
        return results

    def __execute_text_pull(self, cur, job_ids):
        query = "SELECT nlp.Report.ReportNo,nlp.Report.ReportText " \
                "FROM nlp.Report " + \
                "WHERE nlp.Report.Status=\'approved\'" + \
                self.__AND_nlp_Report_JobRunID_IN(job_ids)
        cur.execute(query)
        results = {}
        for row in cur.fetchall():
            results[row[0]] = row[1]
        return results

    def __AND_nlp_Report_JobRunID_IN(self, list_jobids):
        query_str = " AND nlp.Report.JobRunID IN ("
        for num in list_jobids:
            query_str += str(num) + ", "
        query_str = query_str.rstrip(", ")
        query_str += ") "
        return query_str

    def _add_annotation(self, document, annotation):
        start = annotation["StartPosition"]
        end = annotation["StopPosition"]
        text = document.text[start:end]
        sent_idx = None
        tag = annotation["Field"].lower()
        self.detected_annotations.add(tag)
        # Figure out which sentence the tagged text belongs to
        for i, sent in enumerate(document.sentences):
            sent_start = sent.span_start
            sent_chunk = sent.text[start - sent_start:end - sent_start]
            if sent_chunk == text:
                sent_idx = i
                break
        # Store event on doc level
        e = GoldAnnotation(tag, start, end, text, doc_id=document.document_id)
        if tag not in document.concepts_gold:
            document.concepts_gold[tag] = list()
        document.concepts_gold[tag].append(e)
        return True