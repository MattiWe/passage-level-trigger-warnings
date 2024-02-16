from statistics import mode
from copy import deepcopy

warning_map = {
    "Misogyny/ Sexism - Train": 'misogyny', "Misogyny/ Sexism - Test": "misogyny",
    "Racism - Train": 'racism', "Racism - Test": "racism",
    "Ableism - Train": 'ableism', "Ableism - Test": "ableism",
    "Homophobia - Train": 'homophobia', "Homophobia - Test": "homophobia",
    "Death - Train": 'death', "Death - Test": "death",
    "Violence - Train": 'violence', "Violence - Test": "violence",
    "Kidnapping/ Abduction - Train": 'abduction', "Kidnapping/ Abduction - Test": "abduction",
    "War - Train": 'war', "War - Test": "war",
}


major_warning = {
    "misogyny": "discrimination",
    "racism": "discrimination",
    "ableism": "discrimination",
    "homophobia": "discrimination",
    "death": "violence",
    "violence": "violence",
    "abduction": "violence",
    "war": "violence",
}


def _r(value, decimals=2, pct=False):
    v = round(value, decimals)
    if pct:
        v = v * 100
    return v


def multilabel_to_multiclass(multilabel: list, level='major'):
    multilabel = [int(x) for x in multilabel]
    if not 1 in multilabel:
        return False
    idx = multilabel.index(1)
    minor = list(major_warning.keys())[idx]
    if level == 'minor':
        return minor
    return major_warning[minor]


def preprocess(st: str) -> str:
    """ Preprocess the texts
    1. Normalize quotes
    2. remove leading special characters and parsing relics
    3. lowercase
    4. omit very short string (less than three characters)
    """
    st = st.replace("”", "\'").replace("“", "\'").replace("\"", "\'").replace("’", "\'")\
        .replace(" ", " ").replace("\u261e", ">").replace("\u2013", "-").replace("\u2014", "-")\
        .replace("\u2026", "...").replace("\u2018", "\'").replace("\u2022", "-")

    return st.lower().strip().lstrip(".,:;,.\\|/?-_=+!@#$%^&*(){}[]").removeprefix("chapter text")

def infer_label(date, method='majority'):
    """Do label inference from the individual votes

    :param date: a dicts with a "labels" key, which contains a list of votes
    :param method: "majority" or "minority", defaults to 'majority'
    """
    votes = date["labels"]
    if method=='majority':
        inferred_warning = mode(votes) if mode(votes) >= 0 else votes[0]
    elif method=='minority':
        inferred_warning = 1 if 1 in votes else 0
    new_date = deepcopy(date)
    new_date["labels"] = inferred_warning
    return new_date