import re

def strip_comment_line_and_append_line_number(val, line, to_append):
    val = val.split("\n")
    for sval in val:
        sval = strip_comment_line(sval)
        if sval == "":
            continue
        to_append.append((sval, line))
        line = line + 1

def strip_comment_line(val):
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

def seq(x):
    """ Max 10 long sequences, consecutive."""
    if len(x) == 0:
        return []
    res = [(x[0],)]
    nexts = seq(x[1:])
    for n in nexts:
        if n[0] == x[0] + 1 and len(n) < 9:
            res = res + [(x[0],) + n]
    return res + nexts

def comms_to_seq(file):
    l = len(file)
    resp = []
    for i in seq(range(l)):
        coming_from = []
        long_comm = ""
        for ii in i:
            long_comm = long_comm + re.sub(r'[^\w\s]', '', file[ii][0])
            coming_from.append(file[ii][1])
        resp.append((long_comm, coming_from))
    return resp

def comms_to_seq_doc(file, nlp):
    resp = comms_to_seq(file)
    return [(nlp(long_comm), coming_from) for long_comm, coming_from in resp]