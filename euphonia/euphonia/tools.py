import os


def get_file(filename):
    full_name = (
            # os.getcwd() +
            "cache/" + filename + ".mid")

    if os.path.exists(full_name):
        return full_name
    else:
        return