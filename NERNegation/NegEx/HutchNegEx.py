# Copyright (c) 2016-2017 Fred Hutchinson Cancer Research Center
#
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#
import csv
import os

import re


class HutchNegEx(object):
    def __init__(self):
        self.cwd = os.path.dirname(__file__)
        self.negation_patterns = self._load_patterns()
        self.types_we_care_about = {"DEFINITE_NEGATED_EXISTENCE":3, "PROBABLE_NEGATED_EXISTENCE":2, "AMBIVALENT_EXISTENCE":1}

    def negate(self, doc_obj):
        '''
        Given a document object labelled with NER predictions, find negation entities and determine their scope
        :param Document objcet:
        :return: Document object altered with a negated flag on negated Named Entities
        '''
        matched_negs_in_doc = list()
        for s in doc_obj.sentences:
            text = s.text
            matched_negations = self._match_negation(text)
            matched_negations = self._update_to_doc_lvl_spans(s.span_start, matched_negations)
            matched_negs_in_doc.extend(matched_negations)

        self._set_negation_indexes(doc_obj, matched_negs_in_doc)
        self._negate_named_entities(doc_obj)

    def _load_patterns(self):
        with open(os.path.join(self.cwd, "Target_Definitions","clinical_neg_only.tsv")) as csvfile:
            pattern_reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
            column_names=list()
            list_pattern_dicts=list()
            for i, row in enumerate(pattern_reader):
                if i ==0:
                    column_names = row
                elif row[2] != "":
                    list_pattern_dicts.append({
                        column_names[0]:row[0],
                        column_names[1]:row[1],
                        column_names[2]:re.compile('\s*'+row[2]+'(\s+|\.|\?)', re.IGNORECASE),
                        column_names[3]:row[3],
                        column_names[4]:row[4],
                        column_names[5]:row[5],
                        column_names[6]:row[6],
                        column_names[7]:row[7]
                    })
                else:
                    list_pattern_dicts.append({
                        column_names[0]: re.compile('\s*'+row[0]+'(\s+|\.|\?|:)', re.IGNORECASE),
                        column_names[1]: row[1],
                        column_names[2]: row[2],
                        column_names[3]: row[3],
                        column_names[4]: row[4],
                        column_names[5]: row[5],
                        column_names[6]: row[6],
                        column_names[7]: row[7]
                    })
        return list_pattern_dicts

    def _match_negation(self, text):
        negation_matches = list()
        for p in self.negation_patterns:
            if p['Regex'] is not "":
                pat = p['Regex']
                for m in pat.finditer(text):
                    negation_matches.append((m.span(), m.group(), p['Type'], p['Direction']))
            else:
                pat = p['Lex']
                for m in pat.finditer(text):
                    negation_matches.append((m.span(), m.group(), p['Type'], p['Direction']))
        return negation_matches

    def _update_to_doc_lvl_spans(self,sent_span_start_in_doc, matched_negations):
        new_sentence_negations = list()
        for n in matched_negations:
            new_sentence_negations.append(((sent_span_start_in_doc+n[0][0],
                                            sent_span_start_in_doc+n[0][1]),
                                           n[1],
                                           n[2],
                                           n[3],
                                           ))
        return new_sentence_negations

    def _get_tok_start_and_stop_idxs(self, doc_start, doc_stop, doc):
        tokspan_list = doc.token_spans

        # a and b define the range of values to examine from the tokspan list at each layer of recursion
        a = 0
        b = len(tokspan_list)-1
        start_tok = self._find_tok_idx(a, b, tokspan_list, doc_start)
        end_tok = self._find_tok_idx(a, b, tokspan_list, doc_stop)
        return start_tok, end_tok

    def _find_tok_idx(self,a, b, tokspan_list, doc_idx):
            if len(tokspan_list[a:b]) == 1:
                return a
            tok_idx = len(tokspan_list[a:b]) / 2
            tok_idx +=a
            if doc_idx >= tokspan_list[tok_idx][0] and doc_idx <= tokspan_list[tok_idx][1]:
                return tok_idx
            elif doc_idx < tokspan_list[tok_idx][0] and doc_idx < tokspan_list[tok_idx][1]:
                # recurse with the lower half of the list
                return self._find_tok_idx(a, tok_idx, tokspan_list, doc_idx)
            elif doc_idx > tokspan_list[tok_idx][0] and doc_idx > tokspan_list[tok_idx][1]:
                # recurse with the upper half of the list
                return self._find_tok_idx(tok_idx, b, tokspan_list, doc_idx)

    def _set_negation_indexes(self, doc_obj, matched_negs_in_doc):
        negations = list()
        for m in matched_negs_in_doc:
            ent_type = m[2]
            text = m[1]
            action = m[3]
            doc_start = m[0][0]
            doc_stop = m[0][1]
            if ent_type in self.types_we_care_about:
                tok_start_idx, tok_stop_idx = self._get_tok_start_and_stop_idxs(doc_start, doc_stop, doc_obj)
                negations.append((tok_start_idx, tok_stop_idx, ent_type, action, text))
        doc_obj.negation_indexes = negations

    def _negate_named_entities(self, doc_obj):
        negs = doc_obj.negation_indexes
        for n in negs:
            doc_obj.NER_token_labels = self._resolve_scope_of_negation(n, doc_obj.NER_token_labels)

    def _resolve_scope_of_negation(self, negation, label_toks):
        t1 = negation[0]
        t2 = negation[1]
        label = negation[2]
        action = negation[3]
        scope = 4

        if action == "forward":
            self._scope_crawl_forward(t2, scope, label_toks, label)

        if action == "backward":
            self._scope_crawl_backward(t1,scope,label_toks, label)

        if action == "bidirectional":
            self._scope_crawl_forward(t2, scope, label_toks, label)
            self._scope_crawl_backward(t1, scope, label_toks, label)
        return label_toks

    def _scope_crawl_forward(self,t2, scope, label_toks, label):
        window_wideners = {",", ";", ":", "or"}
        window_breakers = {".", "but", "\n", "and","except"} # 'And' in the context of negation almost always serves as
                                                    #  a clausal conjunction, not a continuation of a negated list.
                                                    # 'or' is always used for continuation of a negation list.
        scope = self._recalculate_scope(t2, scope, window_wideners, label_toks)
        for i in range(t2, t2 + scope, 1):
            if i < len(label_toks):
                current_word = label_toks[i]['text']
                if current_word in window_breakers:
                    return
                while i < len(label_toks) and label_toks[i]['label'] != "O":
                    label_toks[i] = self._add_negation_label(label_toks[i], label)
                    i += 1

    def _scope_crawl_backward(self, t1, scope, label_toks, label):
        for i in range(t1, t1 - scope, -1):
            if i < len(label_toks):
                if label_toks[i]['label'] != "O" :
                    while label_toks[i]['label'] != "O":
                        label_toks[i] = self._add_negation_label(label_toks[i], label)
                        i -= 1

    def _create_negation_column_in_result_tuples(self, result_tuples):
        if result_tuples[0][:-1] not in self.types_we_care_about: # Then we havent assigned negation to this tuple yet;
                                                                #  append the dummy label
            new_result_tuples = list()
            for r in result_tuples:
                t_list = list(r)
                t_list.append("Not_Negated")
                new_result_tuples.append(t_list)
            return new_result_tuples
        else:
            return result_tuples

    def _add_negation_label(self, label_tok_dict, label):
        if 'negation' not in label_tok_dict:
            label_tok_dict['negation'] = label
        else:
            if self.types_we_care_about[label] > self.types_we_care_about[label_tok_dict['negation']]:
                label_tok_dict['negation'] = label
        return label_tok_dict

    def _recalculate_scope(self, start, scope, window_wideners, label_toks):
        new_scope = scope
        relevant_tokens = label_toks[start-1:start+scope+15]
        stall=0
        for i, t in enumerate(relevant_tokens):
            if t['text'] in window_wideners:
                new_scope += 2

        return new_scope
