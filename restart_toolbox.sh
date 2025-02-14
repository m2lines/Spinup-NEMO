#! /bin/bash
set -e

# TODO - command line arguments
TEST=DINO
PROCS=36
TIME=00576000

if [ -z "$TEST" ] || [ -z "$PROCS" ] || [ -z "$TIME" ]; then
  echo "Error: One or more required variables (TEST, PROCS, TIME) are not set."
  exit 1
fi

# Path to main_restart.py
# cd to directory where this script is.
cd "$(dirname "$0")"

# Check if main_restart.py exists
if [ ! -f "main_restart.py" ]; then
  echo "Error: main_restart.py not found in the script directory."
  exit 1
fi

# Path to NEMO, XIOS and tools
# This file must be created before the execution of this script (system specific)
source env.sh

# Check if env.sh exists
if [ ! -f "env.sh" ]; then
  echo "Error: env.sh not found containing, REBUILD_NEMO/rebuild_nemo, nemo_output_dir or ml_prediction_dir."
  exit 1
fi

# Check if REBUILD_NEMO is set and executable, OR if `rebuild_nemo` is in PATH
if [ -n "$REBUILD_NEMO" ] && [ -x "$REBUILD_NEMO" ] || command -v rebuild_nemo &>/dev/null; then
  echo "REBUILD_NEMO is set correctly or available in PATH."
else
  echo "Error: REBUILD_NEMO is not set to an executable file and 'rebuild_nemo' is not found in PATH."
  exit 1
fi

# Check if output directories set
if [ -n "$nemo_output_dir" ] && [ -n "$ml_prediction_dir" ]; then
  echo "Output directories set"
else
  echo "One of nemo_output_dir or ml_prediction_dir not correctly set"
  exit 1
fi

# The following files are found in the job directory
# - path to simulation output containing the following:
# - 'mask_file_[00**]' : Mask file which masks out areas where the ocean does not exist, suffixed by processor.
# - '$TEST_[<time>]_[00**]': Checkpoint/restart file, suffixed by processor.

RESTART_NAME=${TEST}_${TIME}_restart


# 2: Combine files
# this combines all files indexed by $PROCS into a single file.
$REBUILD_NEMO -n ./nam_rebuild $nemo_output_dir/$RESTART_NAME $PROCS &> out.txt
$REBUILD_NEMO -n ./nam_rebuild $nemo_output_dir/mesh_mask $PROCS &> out.txt

echo "Files recombined in $output as $nemo_output_dir/$RESTART_NAME.nc and $nemo_output_dir/mesh_mask.nc"

echo "Executing main_restart.py"

# 3: Run main_restart.py to create new restart file
python main_restart.py \
  --restart_path $nemo_output_dir \
  --radical $RESTART_NAME \
  --mask_file $nemo_output_dir/mesh_mask.nc \
  --prediction_path $ml_prediction_dir

# outputs with prefix NEW.

echo "New restart file in $nemo_output_dir as $nemo_output_dir/NEW_${RESTART_NAME}_restart.nc"
