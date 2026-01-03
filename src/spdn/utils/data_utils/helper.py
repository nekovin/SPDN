import re


def extract_number(filename):
    match = re.search(r"\((\d+)\)", filename)
    if match:
        return int(match.group(1))
    return 0
