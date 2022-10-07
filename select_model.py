import numpy as np
import json
import pandas as pd
import glob
import os


def read_file(filename):
    # reading the data from the file
    results = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            results.append(json.loads(line))
        return results


def get_aggregate_results(results):
    aggregate_results = {}
    for result in results:
        if result["model_path"] not in aggregate_results:
            aggregate_results[result["model_path"]] = {
                "input_file": [result["input_file"]],
                "mse": [result["mse"]],
                "psnr": [result["psnr"]],
                # "msssim": [result["msssim"]],
                "bit_rate": [result["bit_rate"]],
            }
        else:
            for metric in [
                "input_file",
                "mse",
                "psnr",
                #    "msssim",
                "bit_rate",
            ]:
                aggregate_results[result["model_path"]][metric].append(result[metric])
    return aggregate_results


def get_average_results(aggregate_results):
    average_results = {}
    for model_path in aggregate_results:
        average_results[model_path] = {}
        for metric in [
            "mse",
            "psnr",
            #    "msssim",
            "bit_rate",
        ]:
            average_results[model_path][metric] = np.mean(
                aggregate_results[model_path][metric]
            )
    return average_results


def print_best_models(aggregate_results, metric="psnr", ascending=False):

    print(f"------------------------------{metric}------------------------------")
    print(f"Best models based on {metric}")
    # Extract the best 5 models based on the average metric
    average_results = get_average_results(aggregate_results)
    df_metrics = pd.DataFrame(average_results).T
    df_metrics = df_metrics.sort_values(by=metric, ascending=ascending)
    best_models = df_metrics[:5].index.values.tolist()
    # print out the best 5 models
    for model in best_models:
        print(model)
        df = pd.DataFrame(aggregate_results[model])
        df["CR"] = 32 / df["bit_rate"]
        print(df)
        print("\n")


if __name__ == "__main__":
    output_dir = "./outputs"
    files = glob.glob(os.path.join(output_dir, "*.txt"))
    # print(files)

    for file in files:
        print(file)
        results = read_file(file)
        aggregate_results = get_aggregate_results(results)
        # print_best_models(aggregate_results, metric="psnr", ascending=False)
        print_best_models(aggregate_results, metric="mse", ascending=True)
        # print_best_models(aggregate_results, metric="bit_rate", ascending=True)
        # print_best_models(aggregate_results, metric="msssim", ascending=False)
        print("")
