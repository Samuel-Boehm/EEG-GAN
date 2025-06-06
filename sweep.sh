#!/bin/bash

# Define the partition on which the job shall run.
#SBATCH -p ml_gpu-rtx2080 # short: -p <partition_name>

# Define a name for your job
#SBATCH --job-name eeg-gan 

# Define the files to write the outputs of the job to.
# Please note the SLURM will not create this directory for you, and if it is missing, no logs will be saved.
# You must create the directory yourself. In this case, that means you have to create the "logs" directory yourself.

#SBATCH --output logs/%x-%A-%a.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A-job_name.out
#SBATCH --error logs/%x-%A-%a.err    # STDERR  short: -e logs/%x-%A-job_name.out

# Define the amount of memory required per node
#SBATCH --mem 32GB
#SBATCH --time 1-12:00:00
echo "Workingdir: $PWD";
echo "Started at $(date)";

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

source ~/miniconda3/bin/activate # Adjust to your path of Miniconda installation
conda activate eeg-gan

# Running the job
start=`date +%s`

srun wandb agent --count 1 $1

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.