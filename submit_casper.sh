#!/bin/bash -l
#PBS -A UGIT0046
#PBS -N fjord
#PBS -j oe
#PBS -r n
#PBS -l walltime=12:00:00
#PBS -q casper
#PBS -l select=1:ncpus=1:mem=40GB:ngpus=1:gpu_type=a100
#PBS -m abe
#
# Run one scenario of fjord.jl on an NCAR Casper A100. Pass the scenario (and any other
# --field=value overrides) through the SCENARIO / EXTRA variables:
#
#   qsub -v SCENARIO=control                         submit_casper.sh
#   qsub -v SCENARIO=tidepump,EXTRA=--stop_days=7    submit_casper.sh
#
# Resumes automatically if a checkpoint for the run name is already in the output dir.

cd "$PBS_O_WORKDIR" || exit 1
mkdir -p logs
module purge
module load ncarenv/23.10

JULIA="${JULIA:-$HOME/.juliaup/bin/julia}"
SCENARIO="${SCENARIO:-control}"
EXTRA="${EXTRA:-}"
OUTDIR="${OUTDIR:-$PBS_O_WORKDIR/output/$SCENARIO}"
mkdir -p "$OUTDIR"

# Casper hands out the GPU by UUID, which CUDA.jl cannot parse; translate it to an index.
if [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* || "${CUDA_VISIBLE_DEVICES:-}" == MIG-* ]]; then
    nvis=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')
    if [[ "$nvis" -le 1 ]]; then
        export CUDA_VISIBLE_DEVICES=0
    else
        idx=$(nvidia-smi --query-gpu=uuid,index --format=csv,noheader \
              | awk -F', *' -v u="$CUDA_VISIBLE_DEVICES" '$1==u{print $2; exit}')
        export CUDA_VISIBLE_DEVICES="${idx:-0}"
    fi
fi

$JULIA --project fjord.jl \
    --name="$SCENARIO" --scenario="$SCENARIO" --architecture=gpu \
    --output_dir="$OUTDIR" $EXTRA \
    2>&1 | tee "logs/${SCENARIO}.out"
