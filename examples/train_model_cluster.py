'''
The script recognizes .
Tom Stuttard
'''

import os
from graphnet.utils.cluster_params import *

# training example, full variable list available in cluster.py
with ClusterSubmitter(
        job_name="test_job",
        
        flush_factor=1,  # How many commands to bunch into a single job [batch array later?]
        num_cpus=2,
        num_gpus=1,
        memory=2*4096, # most clusters have 4gb memory allocated per cpu
        disk_space=1000,
        submit_dir=os.path.join(os.path.expanduser('~'),'graphnet','results','job'),
        output_dir=os.path.join(os.path.expanduser('~'),'graphnet','results','job'),
        run_locally=False,
	#if on npx
        cluster_name="icecube_npx", # negates the npx or grid prompt when running on cobalt
	start_up_commands=os.path.join('source ',os.path.expanduser("~/conda.sh"))
    ) as submitter :

        print("Testing submitter")

        submitter.add(
		command="sh /data/user/mholm/graphnet/examples/train_model_cluster.sh",
		description="test",
		allowed_return_status=[5,0]
	) # Define acceptable return status values

	#submitter.add(
	#	command="source /graphnet/examples/train_model_cluster.sh",
	#	description="test",
	#	allowed_return_status=[5,0]
	#)
	
	#submitter.add(
	#	command=os.path.join("sh ", os.path.expanduser("~/train_model_cluster.sh")),
	#	description="test",
	#	allowed_return_status=[5,0]
	#)
