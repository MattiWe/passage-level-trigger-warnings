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
