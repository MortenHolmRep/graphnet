#!/bin/bash
## -----------------
## Branch: noise_cleaning
## In terminal, run bash as 'source train_noise_cleaning.sh'
## Good practise: use 'gpustat' and select an unused gpu.

# conda work environment activated
source /groups/icecube/${USER}/.bashrc
cd /groups/icecube/${USER}/graphnet_user/env/
conda activate gnn_py38_leon

# database and location
# /groups/icecube/asogaard/data/sqlite/dev_step4_numu_140021_second_run/data/dev_step4_numu_140021_second_run.db

# input: SplitInIcePulse
# truth: SplitInIcePulse_Truthflag - TODO: modifiy sqlite_dataset.py - new function _query_noise_database()
# Benchmark: GraphSagePredictions
# detector: IceCubeUpgrade
# gnn: Dyn_edgeV3 - TODO: test viability and dimension of disabled sections
# target: 'truth_flag' - BinaryClassificationTask

cd /groups/icecube/${USER}/graphnet/cleaning/
#python noise_cleaning.py
python train_model_cleaning.py

conda deactivate
#cd /groups/icecube/${USER}/graphnet_user/