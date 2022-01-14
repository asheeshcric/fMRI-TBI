import os
import csv

def read_score_file(file_path, n_back):
    with open(file_path, "r") as file:
        f_scores = file.readlines()
        f_scores = [int(score.replace("\n", "")) for score in f_scores if score]
    
    return f_scores


def get_scores(sub_path):
    path_0_back = os.path.join(sub_path, "0back_VAS-f.1D")
    path_2_back = os.path.join(sub_path, "2back_VAS-f.1D")
    scores = {
        0: read_score_file(path_0_back, 0),
        2: read_score_file(path_2_back, 2)
    }
    return scores
    

def main():
    data_dir = "/data/fmri/data"
    paths = []
    for sub in os.listdir(data_dir):
        scores = get_scores(os.path.join(data_dir, sub))
        sub_path = os.path.join(data_dir, sub, f"{sub}.preproc")
        for img_name in os.listdir(sub_path):
            img_details = img_name.split(".")
            n_back = int(img_details[0].replace("back", ""))
            n_round = int(img_details[1].replace("r", ""))
            img_path = os.path.join(sub_path, img_name)
            paths.append([img_path, scores[n_back][n_round]])
            
    
    # Write paths and scores to a csv
    filename = "fmri_paths_scores.csv"
    with open(filename, "w") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["img_path", "fatigue_score"])
        csv_writer.writerows(paths)
        


if __name__ == "__main__":
    main()