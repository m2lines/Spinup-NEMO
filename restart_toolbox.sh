#! /bin/bash
set -e
# initial sketch of the restart-toolbox that generates a restart file compatible with NEMO/DINO from:

# Requirements:

# path to simus_predicted
# path to DINO/NEMO simulation output
# path to main_restart.py
# path to REBUILD_NEMO tools.

# TODO: check if absolute path to REBUILD_NEMO has been provided, if not assume below

# TODO: Check if absolute path has been provided to NEMO otherwise assume below

# TODO: number of procs used for sim. (36)

# The following files are found in the job directory

# - path to DINO output containing the following:
# - 'mask_file_[00**]' : Mask file which masks out areas where the ocean does not exist. Suffixed by processor.
# - 'DINO_[<time>]_[00**]': Checkpoint or restart file which is indexed by processor
# I suspect we can easily extract all that we need by just passing in the DINO output directory.

TEST=DINO
# set path to NEMO
NEMO=/rds/project/rds-5mCMIDBOkPU/ma595/nemo/NEMO/
NEMO_4_2_1=$NEMO/nemo_4.2.1
NEMO_trunk=$NEMO/trunk

output=$NEMO_4_2_1/tests/DINO/notebook-data-50/
ml_output=$output/simus_predicted

# Further command line arguments
PROCS=36
TIME=00576000

RESTART_NAME=${TEST}_${TIME}_restart

# - 'simus_predicted' : ML prediction / jump of T, ssh and salinity obtained from the Jumper notebook

# set up environment
module purge
module load rhel8/default-ccl
module load python/3.11.9/gcc/nptrdpll
module load netcdf-fortran/4.6.1/intel/intel-oneapi-mpi/kqukipdf
module load boost/1.85.0/gcc/zouxm6hy

source ./venv3.9/bin/activate

# 1: PATH TO NEMO
export CPATH=$NEMO/trunk/inc:$CPATH
export LD_LIBRARY_PATH=$NEMO_trunk/lib:$LD_LIBRARY_PATH

REBUILD_NEMO=${NEMO_4_2_1}/tools/REBUILD_NEMO/rebuild_nemo
# /rds/project/rds-5mCMIDBOkPU/ma595/nemo/NEMO/nemo_4.2.1/

#2 : Run rebuild_nemo on the mesh_mask files and checkpoint files
$REBUILD_NEMO -n ./nam_rebuild $output/$RESTART_NAME $PROCS &> out.txt
$REBUILD_NEMO -n ./nam_rebuild $output/mesh_mask $PROCS &> out.txt

echo "Files recombined in $output as $output/$RESTART_NAME.nc and $output/mesh_mask.nc"

echo "Executing main_restart.py"

#3 : Run main_restart.py to create new restart file
python main_restart.py \
  --restart_path $output \
  --radical $RESTART_NAME \
  --mask_file $output/mesh_mask.nc \
  --prediction_path $ml_output

# outputs with prefix NEW.

echo "New restart file in $output as $output/NEW_${RESTART_NAME}_restart.nc"
