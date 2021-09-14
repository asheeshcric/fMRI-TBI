import os
import json

data_dir = "/data/fmri/data"


def make_int(num):
    try:
        num = int(num.replace("\n", ""))
        return num
    except ValueError:
        return -1


hc_scores = {"0_back": [], "2_back": []}
tbi_scores = {"0_back": [], "2_back": []}

for subject in os.listdir(data_dir):

    sub_path = os.path.join(data_dir, subject)
    with open(os.path.join(sub_path, "0back_VAS-f.1D"), "r") as z_back_file:
        z_scores = z_back_file.readlines()
        z_scores = [make_int(score) for score in z_scores]

    with open(os.path.join(sub_path, "2back_VAS-f.1D"), "r") as t_back_file:
        t_scores = t_back_file.readlines()
        t_scores = [make_int(score) for score in t_scores]

    if "tbi" in subject:
        tbi_scores["0_back"] += z_scores
        tbi_scores["2_back"] += t_scores
    else:
        hc_scores["0_back"] += z_scores
        hc_scores["2_back"] += t_scores


with open("hc_scores.json", "w") as json_file:
    json.dump(hc_scores, json_file)

with open("tbi_scores.json", "w") as json_file:
    json.dump(tbi_scores, json_file)

