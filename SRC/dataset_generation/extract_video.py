import json
import pandas as pd
import argparse


def filter_sb(sb_data: dict, categories: tuple[str]) -> list[tuple[str, str, str]]:
    pathes = ["\t".join(("video_path", "task_type", "time_stamp", "time", "question"))+"\n"]
    for i in range(len(sb_data)):
        for j in range(len(sb_data[i]["questions"])):
            if sb_data[i]["questions"][j]["task_type"] in categories:
                pathes.append(
                    "\t".join((
                        sb_data[i]["video_path"],
                        sb_data[i]["questions"][j]["task_type"],
                        sb_data[i]["questions"][j]["time_stamp"],
                        sb_data[i]["time"],
                        sb_data[i]["questions"][j]["question"]
                    ))+"\n"
                )
    return pathes

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to MVBench or StreamingBench dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save .tsv file")
    return parser.parse_args()

def main():
    args = parse_args()
    if "MVBench" in args.dataset_path:
        mv_categories = (
            "Action Count",
            "Action Localization",
            "Action Prediction",
            "Action Sequence",
            "Episodic Reasoning",
            "Fine-grained Action",
            "Fine-grained Pose",
            "Object Interaction"
        )

        mv_df = pd.read_csv(args.dataset_path, sep="\t")
        mv_df = mv_df[mv_df["task_type"].isin(mv_categories)]
        mv_df["video_path"] = mv_df["prefix"] + mv_df["video"]
        mv_df = mv_df.drop(['answer', 'candidates', "video", "prefix"], axis="columns")

        mv_df.to_csv(args.output_path, sep="\t", index=False)
    else:
        sb_categories = (
            "Action Recognition",
            "Counting",
            "Prospective Reasoning",
            "Spatial Understanding",
        )

        with open(args.dataset_path, encoding="utf-8") as file:
            sb_data = filter_sb(json.load(file), sb_categories)

        with open(args.output_path, "w", encoding="utf-8") as file:
            file.writelines(sb_data)

if __name__=="__main__":
    main()
