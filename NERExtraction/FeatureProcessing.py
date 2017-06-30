from nltk.stem.porter import *


def word2features(sent, i, w2v_clusters=None):
    word = sent[i].orth_
    postag = sent[i].tag_
    shape = sent[i].shape_
    dep = sent[i].dep_
    cluster = sent[i].cluster
    stemmer = PorterStemmer()

    features = [
        'bias',
        'def_class=' + str(get_def_class(word)),
        'stem=' + stemmer.stem(word),
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
        'shape=' + shape,
        'dep=' + dep,
        'browncluster=' + str(cluster),
        'isMeasurement=' + str(is_measurement(word)),
        'hasProblemForm=' + str(has_problem_form(word))
    ]
    if w2v_clusters:
        wvc_cluster_id = w2v_clusters.cluster_lookup(word)
        features.append('w2vcluster'+ str(wvc_cluster_id))
    else:
        print "WARNING: Not using word2vec cluster features. Does your model support this?"

    if i > 0:
        word1 = sent[i - 1].orth_
        postag1 = sent[i - 1].tag_
        dep1 = sent[i - 1].dep_
        features.extend([
            '-1:featureMetricUnit=' + str(feature_metric_unit(word1)),
            '-1:word.lower=' + word1.lower(),
            '-1:stem=' + stemmer.stem(word1),
            '-1:postag=' + postag1,
            '-1:dep=' + dep1,
            '-1:isMeasurement=' + str(is_measurement(word1)),
            '-1:hasProblemForm=' + str(has_problem_form(word1))
        ])
    else:
        features.append('BOS')

    if i < len(sent) - 1:
        word1 = sent[i + 1].orth_
        postag1 = sent[i + 1].tag_
        dep1 = sent[i + 1].dep_
        features.extend([
            '+1:featureMetricUnit=' + str(feature_metric_unit(word1)),
            '+1:word.lower=' + word1.lower(),
            '+1stem=' + stemmer.stem(word1),
            '+1:postag=' + postag1,
            '+1:dep=' + dep1,
            '+1:isMeasurement=' + str(is_measurement(word1)),
            '+1:hasProblemForm=' + str(has_problem_form(word1))
        ])
    else:
        features.append('EOS')
    return features


def get_sentences_with_subsinfo_from_patients(patients):
    sentences = list()
    for patient in patients:
        for document in patient.doc_list:
            for sent_obj in document.sent_list:
                if (len(sent_obj.gold_events)) > 0:  # if the sentence has an event
                    sentences.append(sent_obj)
    return sentences


def get_sentences_from_patients(patients):
    sentences = list()
    for patient in patients:
        for document in patient.doc_list:
            for sent_obj in document.sent_list:
                sentences.append(sent_obj)
    return sentences


def sent2features(sent, clusters=None):
    return [word2features(sent, i, clusters) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token in sent]


def sent2tokensWLabels(sent):
    return [token for token, postag, label in sent]

def is_measurement(word):
    """
    is_measurement()
    Purpose: Checks if the word is a measurement.
    @param word. A string.
    @return      the matched object if it is a measurement, otherwise None.
   """
    regex = r"^[0-9]*( )?(unit(s)|cc|L|mL|dL)$"
    if re.search(regex, word):
        return True
    else:
        return False

def has_problem_form(word):
    """
    has_problem_form()
    Purpose: Checks if the word has problem form.
    @param word. A string
    @return      the matched object if it has problem form, otheriwse None.
    # >>> has_problem_form('prognosis') is not None
    # True
    # >>> has_problem_form('diagnosis') is not None
    # True
    # >>> has_problem_form('diagnostic') is not None
    # True
    # >>> has_problem_form('arachnophobic') is not None
    # True
    # >>> has_problem_form('test') is not None
    # False
    # >>> has_problem_form('ice') is not None
    False
    """
    regex = r".*(ic|is|oma)$"
    if re.search(regex, word) is not None:
        return True
    else:
        return False


def has_test_form(word):
    """
    has_problem_form()
    Purpose: Checks if the word has problem form.
    @param word. A string
    @return      the matched object if it has problem form, otheriwse None.
    # >>> has_problem_form('prognosis') is not None
    # True
    # >>> has_problem_form('diagnosis') is not None
    # True
    # >>> has_problem_form('diagnostic') is not None
    # True
    # >>> has_problem_form('arachnophobic') is not None
    # True
    # >>> has_problem_form('test') is not None
    # False
    # >>> has_problem_form('ice') is not None
    False
    """
    regex = r"\w+(gram)$"
    if re.search(regex, word) is not None:
        return True
    else:
        return False

def feature_metric_unit(word):
    unit = None
    if is_weight(word):
        unit = 'weight'
    elif is_size(word):
        unit = 'size'
    elif is_volume(word):
        unit = 'volume'
    return unit

def is_volume(word):
    """
    is_volume()
    Purpose: Checks if word is a volume.
    @param word. A string.
    @return      the matched object if it is a volume, otherwise None.
    # >>> is_volume('9ml') is not None
    # True
    # >>> is_volume('10 mL') is not None
    # True
    # >>> is_volume('552 dL') is not None
    # True
    # >>> is_volume('73') is not None
    # False
    # >>> is_volume('ml') is not None
    True
    """
    regex = r"^[0-9]*( )?(ml|mL|dL)$"
    if re.search(regex, word):
        return True
    else:
        return False

def is_weight(word):
    """
    is_weight()
    Purpose: Checks if word is a weight.
    @param word. A string.
    @return      the matched object if it is a weight, otherwise None.
    # >>> is_weight('1mg') is not None
    # True
    # >>> is_weight('10 g') is not None
    # True
    # >>> is_weight('78 mcg') is not None
    # True
    # >>> is_weight('10000 milligrams') is not None
    # True
    # >>> is_weight('14 grams') is not None
    # True
    # >>> is_weight('-10 g') is not None
    # False
    # >>> is_weight('grams') is not None
    # True
    """
    regex = r"^[0-9]*( )?(mg|g|mcg|milligrams|grams)$"
    if re.search(regex, word):
        return True
    else:
        return False

def is_size(word):
    """
    is_size()
    Purpose: Checks if the word is a size.
    @param word. A string.
    @return      the matched object if it is a weight, otheriwse None.
    # >>> is_size('1mm') is not None
    # True
    # >>> is_size('10 cm') is not None
    # True
    # >>> is_size('36 millimeters') is not None
    # True
    # >>> is_size('423 centimeters') is not None
    # True
    # >>> is_size('328') is not None
    # False
    # >>> is_size('22 meters') is not None
    # False
    # >>> is_size('millimeters') is not None
    True
    """
    regex = r"^[0-9]*( )?(mm|cm|millimeters|centimeters)$"
    if re.search(regex, word):
        return True
    else:
        return False


def get_def_class(word):
    """
    get_def_class()
    Purpose: Checks for a definitive classification at the word level.
    @param word. A string
    @return      1 if the word is a test term,
                 2 if the word is a problem term,
                 3 if the word is a treatment term,
                 0 otherwise.
    # >>> get_def_class('eval')
    # 1
    # >>> get_def_class('rate')
    # 1
    # >>> get_def_class('tox')
    # 1
    # >>> get_def_class('swelling')
    # 2
    # >>> get_def_class('mass')
    # 2
    # >>> get_def_class('broken')
    # 2
    # >>> get_def_class('therapy')
    # 3
    # >>> get_def_class('vaccine')
    # 3
    # >>> get_def_class('treatment')
    # 3
    # >>> get_def_class('unrelated')
    # 0
    # """
    test_terms = {
        "eval", "evaluation", "evaluations",
        "sat", "sats", "saturation",
        "exam", "exams",
        "rate", "rates",
        "test", "tests",
        "xray", "xrays",
        "screen", "screens",
        "level", "levels",
        "tox", "biopsy"
    }
    problem_terms = {
        "swelling",
        "wound", "wounds",
        "symptom", "symptoms",
        "shifts", "failure",
        "insufficiency", "insufficiencies",
        "mass", "masses",
        "aneurysm", "aneurysms",
        "ulcer", "ulcers",
        "trauma", "cancer",
        "disease", "diseased",
        "bacterial", "viral",
        "syndrome", "syndromes",
        "pain", "pains"
        "burns", "burned",
        "broken", "fractured",
        "bruising", "bleeding",
        "diarrhea"
    }
    treatment_terms = {
        "therapy",
        "replacement",
        "anesthesia",
        "supplement", "supplemental",
        "vaccine", "vaccines"
        "dose", "doses",
        "shot", "shots",
        "medication", "medicine",
        "treatment", "treatments"
    }
    if word.lower() in test_terms:
        return 1
    elif word.lower() in problem_terms:
        return 2
    elif word.lower() in treatment_terms:
        return 3
    return 0


