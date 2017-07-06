# Copyright (c) 2016-2017 Fred Hutchinson Cancer Research Center
#
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#
import re


class Clusters:
    def __init__(self, cluster_dir):
        self.cluster_dir = cluster_dir
        self.clusters2words, self.words2cluster = self._load_clusters(cluster_dir)

    def _load_clusters(self, filepath):
        if not filepath:
            return None, None
        cluster2words = dict()
        words2cluster=dict()
        with open(filepath, "rb") as f:
            lines = f.readlines()
            for line in lines:
                cid_vec = line.split("\t")
                cid = cid_vec[0]
                words = set(cid_vec[1].split())
                for word in words:
                    words2cluster[word] = cid
                cluster2words[cid] = words
        return cluster2words, words2cluster

    def get_list_clusters(self, clusters, tokens):
        if not clusters:
            return None
        else:
            return [self.cluster_lookup(tokens[i].orth_) for i in range(len(tokens))]

    def cluster_lookup(self, word):
        lower_word = word.lower()
        lower_stripped_word = re.sub(r'[(){}<>,.?/:;]', '', word.lower())
        lower_stripped_zeros_word = re.sub(r'\d', '0', lower_stripped_word)

        if word in self.words2cluster:
            return self.words2cluster[word]
        if lower_word in self.words2cluster:
            return self.words2cluster[lower_word]
        if lower_stripped_word in self.words2cluster:
            return self.words2cluster[lower_stripped_word]
        if lower_stripped_zeros_word in self.words2cluster:
            return self.words2cluster[lower_stripped_zeros_word]
        return None
