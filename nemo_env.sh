#!/bin/bash

# Source submit_run.sh to get the environment variables
source /home/sg2147/Spinup-NEMO/submit_run.sh

#SBATCH -A $PROJECT_ACCOUNT               # Project account for SLURM
#SBATCH -p $PARTITION                     # Partition name (e.g., icelake)
#SBATCH -N $NUM_NODES                     # Number of nodes
#SBATCH -t $TIME_LIMIT                    # Time limit (HH:MM:SS)
#SBATCH -n $NUM_TASKS                     # Number of tasks (processes)

# ======================
# CONFIGURABLE SETTINGS
# ======================
WORK_DIR=${WORK_DIR:-"/path/to/nemo/tests/DINO/EXP00"}
NEMO_EXEC=${NEMO_EXEC:-"nemo"} # Name of the NEMO executable
MODULES=("rhel8/default-ccl" "netcdf-fortran/4.6.1/intel/intel-oneapi-mpi/kqukipdf" "boost/1.85.0/gcc/zouxm6hy")

# Paths for libraries
CPATH=${CPATH:-"/path/to/nemo/trunk/inc"}
LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-"/path/to/nemo/trunk/lib"}

# ======================
# NAVIGATE TO WORK DIR
# ======================
cd "$WORK_DIR" || exit 1

# ======================
# LOAD MODULES
# ======================
module purge
for module in "${MODULES[@]}"; do
    module load "$module"
done

# Set environment variables
export CPATH="$CPATH:$CPATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$LD_LIBRARY_PATH"

# ======================
# CREATE UNIQUE OUTPUT DIRECTORY
# ======================
RUN_ID=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$WORK_DIR/output_$RUN_ID"
mkdir -p "$OUTPUT_DIR"

# ======================
# RUN NEMO SIMULATION
# ======================
mpirun -np "$NUM_TASKS" "$NEMO_EXEC" > "$OUTPUT_DIR/nemo_output.log" 2>&1

# ======================
# MOVE OUTPUT FILES
# ======================
if [ -d "$WORK_DIR/output" ]; then
    mv "$WORK_DIR/output/"* "$OUTPUT_DIR" 2>/dev/null
fi

# Move restart files and prevent overwriting
for file in "$WORK_DIR"/DINO_*_restart.nc; do
    if [ -f "$file" ]; then
        mv "$file" "$OUTPUT_DIR/$(basename "$file" .nc)_$RUN_ID.nc"
    fi
done

echo "Run completed. Outputs stored in: $OUTPUT_DIR"
