#!/bin/bash
#
#------------------Scheduler Options--------------------
#SBATCH -J TDSE                # Job name
#SBATCH -N 1                   # Total number of nodes (16 cores/node)
#SBATCH -n 32                  # Total number of tasks
#SBATCH -p jila                # Queue name
#SBATCH -o run_%j.log          # Name of stdout output file (%j expands to jobid)
#SBATCH -t 12:00:00            # Run time (hh:mm:ss)
#------------------------------------------------------

# To daisy chain jobs, the script must know the name of the job submission script.
# 1) if the $DAISYCHAIN_SCRIPT environment variable is set, that will be used
# 2) by default, the script is set to look in the working directory for a *.slurm file
# The use case is that daisychain jobs will all have their own directory and also 
# their own unique job name.

#----------------------User Options------------------------
TDSE_DIR=/users/becker/jove7731/Repos/TDSE
DAISYCHAIN_SCRIPT=/data/becker/jove7731/daisy_H_High_order_fixed_grid_test/slurm_daisychain.sh
RUN_FILE=${TDSE_DIR}/bin/TDSE
RESTART_FILE=${TDSE_DIR}/scripts/run_scripts/restart_on.py

module purge 
module load intel
module load openmpi
module load hdf5
module load boost
module load cmake
module load blas
module load lapack
module list 
#----------------------------------------------------------

# check if we should already be finished
baseSlurmJobName=$(echo ${SLURM_JOB_NAME} | sed 's/[0-9]*$//g')
if [ -f ${SLURM_SUBMIT_DIR}/daisychain/finished ]; then
  jobDir=${baseSlurmJobName}$(date +%y-%m-%d_%H%M%S)
  mkdir -p ${SLURM_SUBMIT_DIR}/daisychain/history/${jobDir}
  mv ${SLURM_SUBMIT_DIR}/daisychain/* ${SLURM_SUBMIT_DIR}/daisychain/history/${jobDir}
  # and we're done.  We cleaned out the progress files, time to quit
  echo "Found a "finished" file in ${SLURM_SUBMIT_DIR}/daisychain.  Job series is complete, and progress files have been moved to ${SLURM_SUBMIT_DIR}/daisychain/history/${jobDir}"
  exit
fi

# use the filesystem to keep track of our long job
if [ ! -d ${SLURM_SUBMIT_DIR}/daisychain ]; then # we must be the first job
  mkdir ${SLURM_SUBMIT_DIR}/daisychain
  mkdir ${SLURM_SUBMIT_DIR}/daisychain/history
  thisJobNumber=1
else
  lastJobNumber=$(ls ${SLURM_SUBMIT_DIR}/daisychain | egrep "^[0-9]+$" | sort -n | tail -n 1)
  if [ -z $lastJobNumber ]; then
    thisJobNumber=1
  else
    thisJobNumber=$(( lastJobNumber + 1 ))
  fi
fi

# log
touch ${SLURM_SUBMIT_DIR}/daisychain/$thisJobNumber
echo "Slurm Job number $SLURM_JOB_ID entitled ${SLURM_JOB_NAME} started on $(date) on node(s) $SLURM_NODELIST." > ${SLURM_SUBMIT_DIR}/daisychain/$thisJobNumber

# submit dependent job
nextJobNumber=$(( thisJobNumber + 1 ))
nextSlurmJobName=${baseSlurmJobName}$nextJobNumber

echo "Submitting Dependent Job:"
sbatch -J $nextSlurmJobName --dependency=afterany:$SLURM_JOB_ID ${DAISYCHAIN_SCRIPT}



#-------------------Job Goes Here--------------------------
# Job goes here. This could be done multiple ways.  Assuming we need continued
# jobs to be different from the first job, I would frame it out this way:
if [ "$thisJobNumber" -eq "1" ]; then
 #first job
 echo "Starting First Job:"
 mpiexec ${RUN_FILE} -log_view
else
 #continuation
 echo "Starting Continuation Job:"
 python ${RESTART_FILE}
 sleep 1
 mpiexec ${RUN_FILE} -log_view
fi
#----------------------------------------------------------



# if this runs, the job completed:
echo "Daisy Chain Complete"
touch ${SLURM_SUBMIT_DIR}/daisychain/finished

# help catch runaway scripts
sleep 30

