import ConfigParser
import os
import re
from collections import defaultdict

from AbstractClasses import AbstractAnnotation


class Document(object):
    def __init__(self, document_id, text):
        self.concepts_gold = dict()

        self.document_id = document_id
        self.patient_id = document_id.split("_")[0] if document_id.split("_")[0] is not self.document_id else None

        self.sentences = list()
        self.text = text
        self.token_spans = list()
        self.tokens = list()
        self.sections = dict()
        self.NER_token_labels = list()
        self.negation_indexes = list()

        self.is_i2b2 = None

        # todo: evaluate whether loading config for every doc causes a slowdown. alternative would be to load in Extraction object and pass through to each Doc instantiation
        self.color_config_dir = os.path.join(os.path.dirname(__file__), os.path.join("..", "NERResources","NER_Colors.ini"))
        self.colors = self.set_color_config("concept_colors")

    def get_detected_section_names(self):
        '''
        Retrieve all the sections detected in this document
        :return: List of all section name strings detected in this doc
        '''
        section_names = set()
        for label, section_dict in self.sections.items():
            section_names.add(label)
        return list(section_names)

    def get_section_tokens(self, section_list):
        '''
        Return all tokens belonging to a given list of sections. If no section exists, raise error
        :return: all tokens belonging to a given section if exists, else return error
        '''
        assert self.tokens is not None, "Document.tokens have not been set, did you go through a preprocessing step first?"
        section_tokens_list_dict = defaultdict(list)
        for section in section_list:
            if section in self.sections:
                section_data = self.sections[section]
                for section_entry in section_data:
                    section_tokens_list = list()
                    span_start = int(section_entry['start'])
                    span_end = int(section_entry['end'])
                    # TODO: instead of iterating over list, do recursive binary search for index of start and end spans?
                    for i, tok in enumerate(self.token_spans):
                        token_level_data=list()
                        if tok[0] >= span_start and tok[0] < span_end:
                            token_level_data.append(self.tokens[i])
                            #for list_tok_tups in self.NER_token_labels:
                            token_level_data.append(self.NER_token_labels[i]['text'])
                            section_tokens_list.append(token_level_data)
                    section_tokens_list_dict[section].append(section_tokens_list)
        return section_tokens_list_dict


    def collect_named_entity_stats(self):
        '''
        return statistics citing entity types by section
        :return: 
        '''
        raise NotImplementedError

    def get_crf_training_vectors(self, tags):
        #initialize tags
        self.crf_tokens = [x.string for x in self.tokens]
        self.crf_tags = ["O" for x in self.tokens]

        # combine all gold labels to a single mega list
        all_gold = list()
        for t in tags:
            if t in self.concepts_gold:
                all_gold.extend(self.concepts_gold[t])

        if self.is_i2b2:
            self.crf_tags = self._set_crf_training_vectors_i2b2(all_gold, self.crf_tags)
        else:
            self.crf_tags = self._set_crf_training_vectors(all_gold, self.crf_tags)
        return self.crf_tags

    def _set_crf_training_vectors(self, gold_events, ctags):
        tok_counter = 0
        for tok in self.tokens:
            for gold in gold_events:
                curr_start = tok.idx
                curr_end = tok.idx + len(tok)
                if curr_start >= gold.start and curr_end <= gold.stop:
                    ctags[tok_counter] = gold.label
            tok_counter +=1
        return ctags

    def _set_crf_training_vectors_i2b2(self, gold_events, ctags):
        tok_counter = 0
        for sent in self.sentences:
            for tok in sent.tokens:
                for gold in gold_events:
                    curr_start = tok.idx + sent.span_start
                    curr_end = tok.idx + len(tok) + sent.span_start
                    if curr_start >= gold.start and curr_end <= gold.stop:
                        ctags[tok_counter]=gold.label
                tok_counter +=1
        return ctags

    def set_sections(self, list_of_section_dicts):
        self.sections = list_of_section_dicts

    def set_NER_predictions(self, probabilities,model_name):
        # Expand tuple to have span as well as probability
        final_result_dict = self._expand_result_dicts(self.tokens, probabilities)
        self.NER_token_labels=final_result_dict

    def _expand_result_dicts(self, tokenized_doc, probability):
        final_class_and_span = list()
        classified_text = list()
        for idx, tok in enumerate(tokenized_doc):
            # Retrieve top-scoring label and its marginal probability
            maximum_label = max(probability[idx], key=probability[idx].get)
            maximum_prob = probability[idx][maximum_label]
            classified_text.append((tok.orth_, maximum_label))
            if re.match('^\s+$',
                        tok.orth_) and maximum_label is not 'O':  # If a newline or series of newline chars got tagged, thats probably wrong...reset tag to 'O'
                combined = {
                    'text': tokenized_doc[idx].orth_,
                    'label': 'O',
                    'start': self.token_spans[idx][0],
                    'stop' : self.token_spans[idx][1],
                    'confidence' : 1.0
                }
            else:
                combined= {
                    'text': tokenized_doc[idx].orth_,
                    'label': maximum_label,
                    'start': self.token_spans[idx][0],
                    'stop': self.token_spans[idx][1],
                    'confidence' : maximum_prob
                }
            final_class_and_span.append(combined)
        return final_class_and_span

    def doc2html(self):
        # define html tags and possible colors
        tag_types = dict()

        processed_toks = self._docs2tokens()
        html_header = "<!doctype html>\n<html lang=\"en\">\n<head><meta charset=\"utf-8\">" \
                          "</head>"
        strg = html_header + "<body>"
        end_highlight_tag = "</span>"
        in_highlight = False
        for sent in processed_toks:
            for item in sent:
                word = item[0]
                tagset = set([x for x in item[1:] if x != "O"])
                for tag in tagset:
                    if not in_highlight:
                        in_highlight = True
                        if tag in tag_types:
                            color = tag_types[tag]
                        else:
                            color = self._get_color(tag.split("-")[-1])
                        strg += self._get_highlight_begin_tag(tag, color)
                if len(tagset) == 0 and in_highlight:
                    in_highlight = False
                    strg = strg[:-1] + end_highlight_tag + " "
                strg += word + " "
            strg += "<br> "
        return strg.encode('ascii', 'ignore') + "</body>"

    def _docs2tokens(self):
        processed_toks = list()
        tok_count = 0
        for sent in self.sentences:
            sent_toks = list()
            for i, tok in enumerate(sent.tokens):
                orth = ""
                try:
                    orth = tok.orth_
                except:
                    orth = tok

                paralell_items = list()
                paralell_items.append(orth)

                for model_name, token_tups in self.NER_token_labels:

                    label = token_tups[tok_count][1] + "__"
                    try:# append label and neg status where available
                        paralell_items.append(label+token_tups[tok_count][5])
                    except: #otherwise, just append label
                        paralell_items.append(label.rstrip("__"))

                sent_toks.append(paralell_items)
                tok_count +=1
            processed_toks.append(sent_toks)
        return processed_toks

    def _get_color(self, curr_tag):
        tag_items = curr_tag.split("__")
        if len(tag_items)==2:
            curr_tag = tag_items[0]
            neg_tag = tag_items[1]
            if curr_tag.lower() in self.colors and neg_tag.lower() in self.colors:
                return ";".join([self.colors[curr_tag.lower()], self.colors[neg_tag.lower()]])

        else:
            if curr_tag.lower() in self.colors:
                return self.colors[curr_tag.lower()]
            else:
                raise Exception("Uh oh. This gold label doesnt exist in our color dictionary.")

    def _get_highlight_begin_tag(self,tag, color):
        return "<span type=\""+tag+"\" style=\"" + color + "\">"

    def set_color_config(self, color_config_section):
        dir = self.color_config_dir
        config = ConfigParser.ConfigParser()
        config.read(dir)
        if config.has_section(color_config_section):
            config_dict = dict()
            options= config.options(section=color_config_section)
            for op in options:
                curr_val = config.get(color_config_section, op)
                config_dict[op]=curr_val
            return config_dict
        else:
            raise Exception("Config has no section as specified to the DocumentPrinter class: check that your desired "
                            "section exists in the NERUtilities/NER_Colors.ini")


    def _get_model_colors(self):
        return self.colors.copy()



class Sentence(object):
    def __init__(self, text, spanstart, spanend, tokens):
        self.text = text
        self.span_start = spanstart
        self.span_end = spanend
        self.tokens = tokens
        self.shape_tokens = [tokens[i].shape_ for i in range(len(tokens))]
        self.pos_tags = [tokens[i].pos_ for i in range(len(tokens))]
        self.dependency_tags = [tokens[i].dep_ for i in range(len(tokens))]
        self.dependency_tags_heads = [tokens[i].head.orth_ for i in range(len(tokens))]
        self.entity_label = [tokens[i].ent_type_ for i in range(len(tokens))]


class GoldAnnotation(AbstractAnnotation):
    def __init__(self, label, start, stop, text, sent_idx):
        super(GoldAnnotation, self).__init__(label, start, stop, text, sent_idx)
        self.prediction_confidence=1


class PredictedAnnotation(AbstractAnnotation):
    def __init__(self, label, start, stop, text, doc_id, prediction_confidence=None):
        super(PredictedAnnotation, self).__init__(label, start, stop, text, doc_id)
        self.prediction_confidence = prediction_confidence