"""The script recognizes .

Tom Stuttard
"""

import os
from graphnet.utilities.cluster import *
from graphnet.utilities.cluster_params import ClusterSubmitter

# training example, full variable list available in cluster.py
with ClusterSubmitter(  # type: ignore
    job_name="test_job",
    flush_factor=1,  # How many commands to bunch into a single job [batch array later?]
    num_cpus=10,
    num_gpus=1,
    memory=10 * 4096,  # most clusters have 4gb memory allocated per cpu
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
            os.path.expanduser("/home/mholm/miniconda3/etc/profile.d/conda.sh"),
        )
    ],
) as submitter:  # type: ignore

    print("Testing submitter")

    submitter.add(
        command=[
            os.path.join(
                "python ",
                os.path.expanduser("/home/mholm/work/graphnet/examples/train_model.py"),
            )
        ],
        description="test",
        allowed_return_status=[5, 0],
    )  # Define acceptable return status values

# submitter.add(
# 	command="source /graphnet/examples/train_model_cluster.sh",
# 	description="test",
# 	allowed_return_status=[5,0]
# )

# submitter.add(
# 	command=os.path.join("sh ", os.path.expanduser("~/train_model_cluster.sh")),
# 	description="test",
# 	allowed_return_status=[5,0]
# )
