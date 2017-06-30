from abc import abstractmethod


class AbstractDataLoader(object):
    def __init__(self):
        self.detected_labels = set() # a Set<String>
        pass

    def get_detected_labels(self):
        '''
        Public-facing getter of detected labels set
        :return: the set of labels detected in the provided annotations
        '''
        return self.detected_labels

    @abstractmethod
    def preprocess(self, spacy_model):
        pass

    @abstractmethod
    def load_documents(self):
        pass

    @abstractmethod
    def load_annotations(self):
        pass


class AbstractAnnotation(object):
    def __init__(self, label, start, stop, text, sent_idx):
        self.label = label
        self.start = start
        self.stop = stop
        self.sent_idx = sent_idx
        self.text = text

    def get_label(self):
        return self.label

    def get_start(self):
        return self.start

    def get_stop(self):
        return self.stop