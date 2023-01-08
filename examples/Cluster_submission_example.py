"""The script recognizes .

Tom Stuttard
"""

import os
from graphnet.utilities.cluster import *
from graphnet.utilities.cluster_params import ClusterSubmitter

# training example, full variable list available in cluster.py
with ClusterSubmitter(  # type: ignore
    job_name="test_job",
    flush_factor=1,  # How many commands to bunch into a single job
    num_cpus=1, #TODO: fetch cpu from training script
    num_gpus=1, #TODO: fetch gpu from training script
    memory=1 * 4096,  # most clusters have 4gb memory allocated per cpu #TODO: dynamic memory sizing according to cpu
    disk_space=1000,
    submit_dir=os.path.join(
        os.path.expanduser("~"), "graphnet", "results", "job"
    ),
    output_dir="/data/user/mholm/storage/condor_results/",
    run_locally=False,
    # if on npx
    cluster_name="icecube_npx", # negates the npx or grid prompt when running on cobalt
    start_up_commands=[
        os.path.join(
            "source ",
            "/home/mholm/miniconda3/etc/profile.d/conda.sh",
        )
    ],
) as submitter:  # type: ignore

    print("Testing submitter")

    submitter.add(
        command=[
            os.path.join(
                "python ",
                "/home/mholm/work/graphnet/examples/train_model.py",
            )
        ],
        description="test",
        allowed_return_status=[5, 0],
    )  # Define acceptable return status values
