#!/bin/bash
# Iterative rerun script with proper job completion checking and queue verification
# Usage: ./rerun.sh <timestamp>
#
# CRITICAL: This script checks both:
# 1. Incomplete experiments (epoch < 1000)
# 2. Experiments already running in SLURM queue (to prevent duplicates)

TIMESTAMP="${1:-20260113_002155}"
MAX_ITERATIONS=10
CHECK_INTERVAL=300  # Check every 5 minutes

echo "========================================================================"
echo "ITERATIVE RERUN WITH JOB MONITORING AND QUEUE VERIFICATION"
echo "Timestamp: ${TIMESTAMP}"
echo "========================================================================"

# Function to get currently running job IDs for this experiment
get_running_jobs() {
    local ts=$1
    # Query squeue for jobs containing the timestamp in their output path
    squeue -u $(whoami) -h -o "%.18i %.30j" 2>/dev/null | awk '{print $1}' | xargs -I {} bash -c "sacct -j {} -o JobID%20 2>/dev/null | grep -q . && echo {}" 2>/dev/null || true
}

# Function to get list of currently running experiments from squeue
get_running_experiments() {
    local ts=$1
    # Get all running jobs with timestamps in their name or path
    squeue -u $(whoami) -h -o "%.30j" 2>/dev/null | grep -E "(XuSparseTPR|SparseTPR|SparseGPR)" || echo ""
}

# Initial check
echo "Initial check of experimental results..."
bash scripts/check_final_epochs.sh ${TIMESTAMP}

echo ""
echo "Checking for currently running experiments in SLURM queue..."
RUNNING=$(get_running_experiments ${TIMESTAMP})
if [ -z "${RUNNING}" ]; then
    echo "✓ No running experiments detected in queue"
else
    echo "⚠️  Currently running experiments detected:"
    echo "${RUNNING}"
    echo ""
    echo "WARNING: If you proceed, duplicate jobs may be submitted for running experiments."
    echo "This can cause data corruption and resource conflicts."
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. Please cancel running jobs first with: scancel -u $(whoami)"
        exit 1
    fi
fi

# Loop until all complete
for iteration in $(seq 1 ${MAX_ITERATIONS}); do
    echo ""
    echo "========================================================================"
    echo "ITERATION ${iteration}/${MAX_ITERATIONS}"
    echo "========================================================================"
    
    # Count incomplete experiments
    INCOMPLETE=$(bash scripts/check_final_epochs.sh ${TIMESTAMP} 2>/dev/null | grep -c "⚠️" || echo "0")
    
    if [ "${INCOMPLETE}" -eq 0 ]; then
        echo "✓ All experiments complete! No incomplete runs found."
        break
    fi
    
    echo "Found ${INCOMPLETE} incomplete experiments."
    
    # Check SLURM queue before generating reruns
    echo ""
    echo "Checking SLURM queue for running experiments..."
    RUNNING_COUNT=$(squeue -u $(whoami) -h 2>/dev/null | wc -l)
    
    if [ ${RUNNING_COUNT} -gt 0 ]; then
        echo "⚠️  WARNING: ${RUNNING_COUNT} job(s) currently in SLURM queue!"
        echo ""
        echo "Job details:"
        squeue -u $(whoami) -h -o "%.18i %.30j %.8T %.10M"
        echo ""
        echo "SAFETY MEASURE: To prevent duplicate submissions,"
        echo "waiting for all existing jobs to complete..."
        
        # Wait for all jobs to complete
        while [ $(squeue -u $(whoami) -h 2>/dev/null | wc -l) -gt 0 ]; do
            REMAINING=$(squeue -u $(whoami) -h 2>/dev/null | wc -l)
            echo "  $(date '+%Y-%m-%d %H:%M:%S') - Waiting for ${REMAINING} job(s) to finish... (check every ${CHECK_INTERVAL}s)"
            sleep ${CHECK_INTERVAL}
        done
        
        echo "✓ All jobs completed!"
        
        # Give time for filesystem sync
        echo "Waiting for filesystem sync..."
        sleep 10
        
        # Re-check incomplete status after jobs complete
        echo ""
        echo "Re-checking results after job completion..."
        bash scripts/check_final_epochs.sh ${TIMESTAMP}
        
        INCOMPLETE=$(bash scripts/check_final_epochs.sh ${TIMESTAMP} 2>/dev/null | grep -c "⚠️" || echo "0")
        if [ "${INCOMPLETE}" -eq 0 ]; then
            echo "✓ All experiments complete!"
            break
        fi
    fi
    
    echo ""
    echo "Generating rerun scripts for ${INCOMPLETE} incomplete experiments..."
    
    # Generate rerun scripts in consolidate mode
    python scripts/generate_rerun_jobs.py ${TIMESTAMP} --consolidate
    
    # Submit jobs and capture job IDs
    echo "Submitting rerun jobs..."
    RERUN_SCRIPT="scripts/rerun_all_${TIMESTAMP}.sh"
    
    if [ ! -f "${RERUN_SCRIPT}" ]; then
        echo "Error: Rerun script not found: ${RERUN_SCRIPT}"
        exit 1
    fi
    
    # Execute the rerun script and extract job IDs
    bash ${RERUN_SCRIPT} > /tmp/submit_output_$$.txt 2>&1
    
    # Extract job IDs from sbatch output (format: "Submitted batch job <job_id>")
    JOB_IDS=$(grep -oP "Submitted batch job \K\d+" /tmp/submit_output_$$.txt || echo "")
    
    if [ -z "${JOB_IDS}" ]; then
        echo "Warning: No job IDs detected. Jobs may have failed to submit or are using different format."
        echo "Waiting ${CHECK_INTERVAL} seconds before next check..."
        sleep ${CHECK_INTERVAL}
    else
        echo "Submitted $(echo ${JOB_IDS} | wc -w) job(s)"
        echo "Job IDs: $(echo ${JOB_IDS} | tr '\n' ' ')"
        
        # Wait for all jobs to complete
        echo "Monitoring job completion..."
        
        while true; do
            # Check if any jobs are still running
            RUNNING_JOBS=0
            for JOB_ID in ${JOB_IDS}; do
                if squeue -j ${JOB_ID} 2>/dev/null | grep -q ${JOB_ID}; then
                    RUNNING_JOBS=$((RUNNING_JOBS + 1))
                fi
            done
            
            if [ ${RUNNING_JOBS} -eq 0 ]; then
                echo "✓ All submitted jobs completed!"
                break
            fi
            
            echo "  $(date '+%Y-%m-%d %H:%M:%S') - ${RUNNING_JOBS} job(s) still running..."
            sleep ${CHECK_INTERVAL}
        done
    fi
    
    # Give a bit of time for files to sync
    echo "Waiting for filesystem sync..."
    sleep 10
    
    # Check results again
    echo ""
    echo "Checking results after rerun..."
    bash scripts/check_final_epochs.sh ${TIMESTAMP}
    
    rm -f /tmp/submit_output_$$.txt
    
    if [ ${iteration} -eq ${MAX_ITERATIONS} ]; then
        echo ""
        echo "⚠️  Reached maximum iterations (${MAX_ITERATIONS})."
        echo "Some experiments may still be incomplete."
        break
    fi
done

echo ""
echo "========================================================================"
echo "RERUN WORKFLOW COMPLETE"
echo "Final results in: results/${TIMESTAMP}/raw/"
echo "========================================================================"
echo ""
echo "Summary:"
echo "  - Original timestamp: ${TIMESTAMP}"
echo "  - Completed iterations: ${iteration}/${MAX_ITERATIONS}"
echo "  - To verify completion: bash scripts/check_final_epochs.sh ${TIMESTAMP}"
echo "========================================================================"