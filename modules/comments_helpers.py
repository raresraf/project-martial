import re
import spacy
from spacy.tokens import Doc


def strip_comment_line_and_append_line_number(val, line, to_append):
    val = val.split("\n")
    for sval in val:
        sval = strip_comment_line(sval)
        if sval == "":
            continue
        to_append.append((sval, line))
        line = line + 1

def strip_comment_line(val: str) -> str:
    val = val.strip("\t").strip("\n").lstrip("/* ").rstrip("/* ")
    if val == '':
        return ""
    if val == '//':
        return ""
    if val == '/*':
        return ""
    if val == '*/':
        return ""
    return val

def generate_comm_sequences(x):
    """ Max 10 long sequences, consecutive."""
    if len(x) == 0:
        return []
    res = [(x[0],)]
    next = generate_comm_sequences(x[1:])
    for n in next:
        if n[0] == x[0] + 1 and len(n) < 9:
            res = res + [(x[0],) + n]
    return res + next

def comm_to_seq(file):
    """From a file, run generate_comm_sequences to generate all possible combinations of consecutive comments."""
    l = len(file)
    resp = []
    for i in generate_comm_sequences(range(l)):
        coming_from = []
        long_comm = ""
        for ii in i:
            long_comm = long_comm + re.sub(r'[^\w\s]', '', file[ii][0]) + " "
            coming_from.append(file[ii][1])
        resp.append((long_comm, coming_from))
    return resp

def comm_to_seq_doc(file, spacy_core_web) -> list[tuple[Doc, int]]:
    """Similar to comm_to_seq but returns the Doc(commentary) instead of commentary: string."""
    resp = comm_to_seq(file)
    return [(spacy_core_web(long_comm), coming_from) for long_comm, coming_from in resp]