from model import MiniCPMo, MiniCPMV, Qwen3
from benchmark import StreamingBench, OmniMMIBench, MVBench, VizWiz

from pathlib import Path
import json
import argparse

import pandas as pd
import transformers

parser = argparse.ArgumentParser()
parser.add_argument("--type", choices=["yes/no vqa", "proactive", "multichoice vqa"])


def main(model: str, bench: str, output_path: str, data_path: str | None) -> None:
    args = parser.parse_args()
    transformers.set_seed(42, deterministic=True)
    if data_path is None: # in case you want to continue from OUTPUT
        data_path = output_path
    
    if not Path(output_path).exists():
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        json.dump([], open(output_path, "w", encoding="utf-8"))

    match bench:
        case "StreamingBench":
            with open(data_path) as file:
                data = json.load(file)
            # with open(output_path) as file:
            #     data = json.load(file)
            bench = StreamingBench()
            task_type = "vqa"
        case "OmniMMI":
            with open(data_path, "r") as file:
                data = json.load(file)
            if args.type != "proactive":
                task_type = "pa"
                bench = OmniMMIBench(task_type=args.type)
                # output_path="output/yesno_omnimmi.json" if args.type == "yes/no vqa" else "output/multi_omnimmi.json"
                print(args.type, output_path)
            else:
                task_type = "pa"
                bench = OmniMMIBench()
        case "MVBench":
            data = pd.read_csv(data_path, sep="\t")
            task_type = "vqa"
            bench = MVBench(224)
        case "VizWiz":
            with open(data_path, "r") as file:
                data = json.load(file)
            task_type = "vqa"
            bench = VizWiz()
        case _:
            raise NotImplementedError(f"No interface for {bench} implemented")

    match model:
        case "MiniCPMo":
            model = MiniCPMo(task_type=task_type)
        case "MiniCPMV":
            model = MiniCPMV(task_type=task_type)
        case "Qwen3":
            model = Qwen3()
        case _:
            raise NotImplementedError(f"No interface for {model} implemented")

    bench.eval(data, model, output_path)

if __name__=="__main__":
    BENCH, DATA=(
        ("StreamingBench", "/path/to/questions_real.json"),
        ("OmniMMI", "/path/to/ommnimmi.json"),
        ("MVBench", "/path/to/MVBench.tsv"),
        ("VizWiz", "/path/to/VizWiz/Annotations/val.json")
    )[0]
    MODEL="MiniCPMo"
    # MODEL="MiniCPMV"
    # MODEL="Qwen3"

    OUTPUT="output/benchmark.json"

    main(MODEL, BENCH, OUTPUT, DATA)

