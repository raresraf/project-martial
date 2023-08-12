import re
from spacy.tokens import Doc
import numpy as np

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

def generate_comm_sequences(x, t):
    """ Max {t} long sequences, consecutive. (usually t = 10)"""
    if len(x) == 0:
        return []
    res = [(x[0],)]
    next = generate_comm_sequences(x[1:], t)
    for n in next:
        if n[0] == x[0] + 1 and len(n) < t - 1:
            res = res + [(x[0],) + n]
    return res + next


def comm_to_seq_default(file, t):
    """From a file, run generate_comm_sequences to generate all possible combinations of consecutive comments."""
    l = len(file)
    resp = []
    for i in generate_comm_sequences(range(l), t):
        coming_from = []
        long_comm = ""
        for ii in i:
            long_comm = long_comm + re.sub(r'[^\w\s]', '', file[ii][0]) + " "
            coming_from.append(file[ii][1])
        resp.append((long_comm, coming_from))
    return resp



def comm_to_seq_elmo(file, elmo) -> list[tuple[Doc, int]]:
    """Similar to comm_to_seq but returns the Doc(commentary) instead of commentary: string."""
    resp = comm_to_seq_default(file, 10)
    ret = []
    for long_comm, coming_from in resp:
        long_comm_tensor = elmo.get_elmo_vectors(long_comm, layers="average")
        long_comm_tensor_avged = np.sum(long_comm_tensor[0][:], axis = 0)/long_comm_tensor.shape[1]
        ret.append((long_comm, coming_from, long_comm_tensor_avged.reshape(1, -1)))
    return ret
