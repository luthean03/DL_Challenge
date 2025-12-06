#!/usr/bin/python

import os
import sys
import subprocess
import tempfile


def makejob(commit_id, configpath, command, nruns, existing_logdirs=None):
    exclude_logs = "--exclude logs" if command == "train" else ""
    existing_dirs_cmd = ""
    if existing_logdirs:
        mkdirs = "\n".join(f"mkdir -p $TMPDIR/code/logs/{d}" for d in existing_logdirs)
        existing_dirs_cmd = f"mkdir -p $TMPDIR/code/logs\n{mkdirs}\n"
    return f"""#!/bin/bash

#SBATCH --job-name=templatecode
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=2:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=1-{nruns}

current_dir=`pwd`
export PATH=$PATH:~/.local/bin

echo "Session " ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}

echo "Running on " $(hostname)

echo "Copying the source directory and data"
date
mkdir $TMPDIR/code
rsync -r --exclude logslurms --exclude configs {exclude_logs} . $TMPDIR/code
{existing_dirs_cmd}

echo "Checking out the correct version of the code commit_id {commit_id}"
cd $TMPDIR/code
git checkout {commit_id}


echo "Setting up the virtual environment"
python3 -m venv venv
source venv/bin/activate

# Install the library
python -m pip install .

echo "Running {command}"
python -m torchtmpl.main {configpath} {command}

if [[ $? != 0 ]]; then
    exit -1
fi

# Copy the generated logs back to the submit directory so they survive the job
rsync -a "$TMPDIR/code/logs/" "$current_dir/logs/"
"""


def submit_job(job):
    with open("job.sbatch", "w") as fp:
        fp.write(job)
    os.system("sbatch job.sbatch")


# Ensure all the modified files have been staged and commited
# This is to guarantee that the commit id is a reliable certificate
# of the version of the code you want to evaluate
result = int(
    subprocess.run(
        "expr $(git diff --name-only | wc -l) + $(git diff --name-only --cached | wc -l)",
        shell=True,
        stdout=subprocess.PIPE,
    ).stdout.decode()
)
if result > 0:
    print(f"We found {result} modifications either not staged or not commited")
    raise RuntimeError(
        "You must stage and commit every modification before submission "
    )

commit_id = subprocess.check_output(
    "git log --pretty=format:'%H' -n 1", shell=True
).decode()

print(f"I will be using the commit id {commit_id}")

# Ensure the log directory exists
os.system("mkdir -p logslurms")

if len(sys.argv) not in [3, 4]:
    print(f"Usage : {sys.argv[0]} config.yaml <train|test> [nruns|1]")
    sys.exit(-1)

configpath = sys.argv[1]
command = sys.argv[2]
if command not in {"train", "test"}:
    raise ValueError("Command must be either 'train' or 'test'")

if len(sys.argv) == 3:
    nruns = 1
else:
    nruns = int(sys.argv[3])

job_configpath = configpath
existing_logdirs = []
if command == "train":
    os.system("mkdir -p configs")
    tmp_configfilepath = tempfile.mkstemp(dir="./configs", suffix="-config.yml")[1]
    os.system(f"cp {configpath} {tmp_configfilepath}")
    job_configpath = tmp_configfilepath
    log_root = "logs"
    if os.path.isdir(log_root):
        existing_logdirs = sorted(
            name
            for name in os.listdir(log_root)
            if os.path.isdir(os.path.join(log_root, name))
        )

# Launch the batch jobs
submit_job(makejob(commit_id, job_configpath, command, nruns, existing_logdirs))