import re


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
    """ Max {t} long sequences, consecutive. (usually t = 6)"""
    total_len = len(x)
    if total_len == 0:
        return []
    res = []
    for i in range(total_len):
        res += [(x[i],)]
        last_res = (x[i], )
        for tt in range(1, t):
            if i + tt >= total_len:
                break
            last_res = last_res + (x[i + tt],)
            res = res + [last_res]
    return res


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



