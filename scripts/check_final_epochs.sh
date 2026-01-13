#!/bin/bash
# Script to examine the final epoch recorded for each CSV file in an experimental run
# Usage: ./check_final_epochs.sh <timestamp>
# Example: ./check_final_epochs.sh 20260113_002155

if [ $# -ne 1 ]; then
    echo "Usage: $0 <timestamp>"
    echo "Example: $0 20260113_002155"
    exit 1
fi

TIMESTAMP=$1
RESULTS_DIR="results/${TIMESTAMP}/raw"

if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Directory $RESULTS_DIR does not exist"
    exit 1
fi

OUTPUT_FILE="results/${TIMESTAMP}/final_epochs_summary_${TIMESTAMP}.txt"

echo "Examining CSV files in $RESULTS_DIR"
echo "=========================================================================================================="
echo ""

# Header
printf "%-50s %-15s %-20s %-5s %-6s %-12s %-12s\n" "File" "Model" "Dataset" "Split" "Epoch" "RMSE" "NLL"
echo "----------------------------------------------------------------------------------------------------------"

# Process each CSV file
for csv_file in "$RESULTS_DIR"/*.csv; do
    if [ -f "$csv_file" ]; then
        filename=$(basename "$csv_file")
        
        # Get the last line (final epoch), skip header
        last_line=$(tail -n 1 "$csv_file")
        
        # Parse CSV fields
        IFS=',' read -r epoch loss elbo log_prior time rmse nll model dataset split <<< "$last_line"
        
        # Print formatted output
        printf "%-50s %-15s %-20s %-5s %-6s %-12s %-12s\n" \
            "$filename" "$model" "$dataset" "$split" "$epoch" "$rmse" "$nll"
    fi
done | tee "$OUTPUT_FILE"

echo ""
echo "=========================================================================================================="
echo "Results saved to: $OUTPUT_FILE"
echo ""

# Count files by model
echo "Summary by Model:"
echo "----------------------------------------------------------------------------------------------------------"
for model in $(tail -n +2 "$OUTPUT_FILE" | awk '{print $2}' | sort -u); do
    count=$(grep -wc "$model" "$OUTPUT_FILE")
    avg_epoch=$(grep -w "$model" "$OUTPUT_FILE" | awk '{sum+=$5; count++} END {if(count>0) print sum/count}')
    avg_rmse=$(grep -w "$model" "$OUTPUT_FILE" | awk '{sum+=$6; count++} END {if(count>0) print sum/count}')
    avg_nll=$(grep -w "$model" "$OUTPUT_FILE" | awk '{sum+=$7; count++} END {if(count>0) print sum/count}')
    
    echo "$model: $count files, avg epoch: $avg_epoch, avg RMSE: $avg_rmse, avg NLL: $avg_nll"
done

# Check for incomplete runs
echo ""
echo "Checking for incomplete runs (epoch < 1000):"
echo "----------------------------------------------------------------------------------------------------------"
tail -n +2 "$OUTPUT_FILE" | awk '$5 < 1000 {print "  ⚠️  " $1 ": epoch " $5}'

incomplete_count=$(tail -n +2 "$OUTPUT_FILE" | awk '$5 < 1000' | wc -l)
if [ "$incomplete_count" -eq 0 ]; then
    echo "  ✓ All runs completed 1000 epochs"
fi
