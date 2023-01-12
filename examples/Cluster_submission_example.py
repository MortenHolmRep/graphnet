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
    num_cpus=13, #TODO: fetch cpu from training script
    num_gpus=1, #TODO: fetch gpu from training script
    memory=13 * 4096,  # most clusters have 4gb memory allocated per cpu #TODO: dynamic memory sizing according to cpu
    disk_space=1000,
    submit_dir=os.path.join(
        os.path.expanduser("~"), "graphnet", "results", "job"
    ),
    output_dir="/data/user/moust/storage/condor_results/",
    run_locally=False,
    # if on npx
    cluster_name="icecube_npx", # negates the npx or grid prompt when running on cobalt
    start_up_commands=[
        ("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/moust/miniconda3/lib/"),
        ("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/moust/miniconda3/envs/graphnet/lib/"),
        ("source /home/moust/miniconda3/etc/profile.d/conda.sh"),
        ("conda activate graphnet")
    ],
) as submitter:  # type: ignore

    print("Testing submitter")

    submitter.add(
        command=[
                ("python /home/moust/work/nmo_analysis/train_upgrade_sweep.py"),
        ],
        description="test",
        allowed_return_status=[5, 0],
    )  # Define acceptable return status values
