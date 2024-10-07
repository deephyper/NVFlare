import glob
import json
import os
import pathlib
import shutil
import subprocess

import pandas as pd
import tensorflow as tf
from deephyper.evaluator import Evaluator
from deephyper.hpo import CBO, HpProblem

### Utility functions to create nvflare job files ###


def read_json(filename):
    assert os.path.isfile(filename), f"{filename} does not exist!"

    with open(filename, "r") as f:
        return json.load(f)


def write_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def update_item_args_in_list(item_list: list, item_id: str, args: dict):
    for item in item_list:
        if item["id"] == item_id:
            item["args"].update(args)


def update_nvflare_job_config(job_dir: str, job_name: str, parameters: dict):

    # Set source path of config files
    client_config_filename = os.path.join(job_dir, job_name, "config", "config_fed_client.json")
    server_config_filename = os.path.join(job_dir, job_name, "config", "config_fed_server.json")
    meta_config_filename = os.path.join(job_dir, "meta.json")

    # Read config files
    client_config = read_json(client_config_filename)
    server_config = read_json(server_config_filename)
    meta_config = read_json(meta_config_filename)
    print(f"Loaded nvflare configs in {job_dir}")

    # Update train_split root
    # TODO: check if this should be the same for all hpo evaluations
    split_dir = os.path.join(job_dir, f"dataset_train_split")
    print(f"Set train split root to {split_dir}")

    server_config["TRAIN_SPLIT_ROOT"] = split_dir
    client_config["TRAIN_SPLIT_ROOT"] = split_dir

    # Update hyperparameters
    update_item_args_in_list(client_config["components"], "cifar10-learner", {"lr": parameters.get("lr", 0.01)})

    # Save updated configs
    write_json(client_config, client_config_filename)
    write_json(server_config, server_config_filename)
    write_json(meta_config, meta_config_filename)
    print(f"Updated {meta_config_filename}")


def get_hp_problem():

    problem = HpProblem()
    problem.add_hyperparameter((1e-5, 1.0, "log-uniform"), "lr", default_value=0.01)

    return problem


def read_eventfile(filepath, tags=["val_acc_global_model"]):
    data = {}
    for summary in tf.compat.v1.train.summary_iterator(filepath):
        for v in summary.summary.value:
            if v.tag in tags:
                # print(v.tag, summary.step, v.simple_value)
                if v.tag in data.keys():
                    data[v.tag].append([summary.step, v.simple_value])
                else:
                    data[v.tag] = [[summary.step, v.simple_value]]
    return data


def add_eventdata(data, filepath, tag="val_acc_global_model"):
    event_data = read_eventfile(filepath, tags=[tag])

    assert len(event_data[tag]) > 0, f"No data for key {tag}"
    # print(event_data)
    for e in event_data[tag]:
        # print(e)
        data["Step"].append(e[0])
        data["Accuracy"].append(e[1])
    print(f"added {len(event_data[tag])} entries for {tag}")


def check_nvflare_job_results(nvflare_workspace_dir):
    data = {"Step": [], "Accuracy": []}

    # add event files
    eventfile = glob.glob(os.path.join(nvflare_workspace_dir, "**", "app_site-1", "events.*"), recursive=True)
    assert len(eventfile) == 1, f"No unique event file found in {nvflare_workspace_dir}!"
    eventfile = eventfile[0]
    print("adding", eventfile)
    add_eventdata(data, eventfile, tag="val_acc_global_model")

    data = pd.DataFrame(data)
    print("Training TB data:")
    print(data)

    return data


def run_fl(job, log_dir="."):

    # Get id of current DeepHyper job
    job_id = int(job.id.split(".")[-1])
    job_dir = os.path.join(log_dir, "jobs", f"{job_id:03d}")
    job_dir = os.path.abspath(job_dir)
    stdoud_file = os.path.join(job_dir, "stdout.txt")
    stderr_file = os.path.join(job_dir, "stderr.txt")

    # Create directory
    pathlib.Path(job_dir).mkdir(parents=True, exist_ok=True)

    # Create nvflare configs (json) using input hyperparameters
    # TODO
    nvflare_job_name = "cifar10_fedavg"
    nvflare_job_dir = os.path.join(job_dir, "nvflare", "job")
    shutil.copytree(f"template_jobs/{nvflare_job_name}", nvflare_job_dir)

    update_nvflare_job_config(
        job_dir=nvflare_job_dir,
        job_name=nvflare_job_name,
        parameters=job.parameters,
    )

    # Run NVFLARE
    with open(stdoud_file, "w") as fsdout, open(stderr_file, "w") as fsderr:

        # Following lines can be uncommented for sanity check
        # result = subprocess.run(
        #     ['echo "Current directory: ${PWD}"'],
        #     stdout=fsdout,
        #     stderr=fsderr,
        #     shell=True,
        # )
        # print(f"{result}")

        threads = 2
        n_clients = 2
        nvflare_workspace_dir = os.path.join(job_dir, "nvflare", "workspace")
        command = (
            f"nvflare simulator {nvflare_job_dir}"
            f" --workspace {nvflare_workspace_dir} --threads {threads} --n_clients {n_clients}"
        )
        print(f"Running: {command}")

        result = subprocess.run(
            [
                command,
            ],
            stdout=fsdout,
            stderr=fsderr,
            shell=True,
        )

    # Collect results and return objective
    df_results = check_nvflare_job_results(nvflare_workspace_dir)

    objective = float(df_results["Accuracy"].iloc[-1])
    metadata = {
        "step": df_results["Step"].tolist(),
        "accuracy": df_results["Accuracy"].tolist(),
    }

    return {"objective": objective, "metadata": metadata}


def test_run_fl():
    from deephyper.evaluator import RunningJob

    problem = get_hp_problem()
    default_parameters = problem.default_configuration
    print(f"{default_parameters=}")

    job = RunningJob(id=0, parameters=default_parameters)
    output = run_fl(job, log_dir="hpo-logs")

    print(f"{output=}")


def main():

    log_dir = "hpo-logs"
    problem = get_hp_problem()
    print(problem)

    evaluator = Evaluator.create(run_fl, method="serial")

    search = CBO(problem, evaluator, log_dir=log_dir)
    results = search.search(max_evals=100)


if __name__ == "__main__":
    # test_run_fl()
    main()
