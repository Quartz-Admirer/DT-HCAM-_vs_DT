#!/usr/bin/env bash

ENV_ID="MiniGrid-MemoryS17Random-v0"
ENV_SHORT_NAME="memorys17random"
DATA_DIR="data/minigrid_memory"
RANDOM_DATA_PATH="${DATA_DIR}/random_${ENV_SHORT_NAME}_trajectories.npz"
FILTERED_DATA_PATH="${DATA_DIR}/medium_filtered_${ENV_SHORT_NAME}_trajectories.npz"
FILTER_THRESHOLD=0.01

echo "--- Configuration ---"
echo "ENV_ID: $ENV_ID"
echo "ENV_SHORT_NAME: $ENV_SHORT_NAME"
echo "DATA_DIR: $DATA_DIR"
echo "Random Data Path: $RANDOM_DATA_PATH"
echo "Filtered Data Path: $FILTERED_DATA_PATH"
echo "---------------------"
sleep 1

echo "[Step 1] Checking for random data file..."
echo "Expected path: '$RANDOM_DATA_PATH'"
echo "Listing directory '$DATA_DIR' contents before check:"
ls -l "$DATA_DIR" || echo "Directory $DATA_DIR not found or ls failed."
echo "Running check: [ ! -f \"$RANDOM_DATA_PATH\" ]"

if [ ! -f "$RANDOM_DATA_PATH" ]; then
    echo "Check result: File NOT found (or check failed). Proceeding with generation..."
    python -m data_generation.generate_minigrid_random \
        --env_id "$ENV_ID" \
        --num_episodes 10000 \
        --max_steps 300 \
        --save_dir "$DATA_DIR" \
        --seed 42

    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "ERROR: Python data generation script failed with exit code $exit_code."
        exit 1
    fi
    echo "Python generation script finished. Checking if file exists NOW..."
    sleep 1
    ls -l "$RANDOM_DATA_PATH"
    if [ ! -f "$RANDOM_DATA_PATH" ]; then
         echo "ERROR: File '$RANDOM_DATA_PATH' STILL does not exist after generation attempt!"
         exit 1
    else
         echo "File '$RANDOM_DATA_PATH' successfully created."
    fi
else
    echo "Check result: File FOUND."
    echo "Random data file already exists: $RANDOM_DATA_PATH"
fi
echo "--- End Step 1 ---"
sleep 1

echo "[Step 2] Checking for filtered data file..."
echo "Expected path: '$FILTERED_DATA_PATH'"
echo "Listing directory '$DATA_DIR' contents before check:"
ls -l "$DATA_DIR" || echo "Directory $DATA_DIR not found or ls failed."
echo "Running check: [ ! -f \"$FILTERED_DATA_PATH\" ]"

if [ ! -f "$FILTERED_DATA_PATH" ]; then
    echo "Check result: Filtered file NOT found. Proceeding with filtering..."
    echo "Checking if source file exists for filtering: '$RANDOM_DATA_PATH'"
    if [ -f "$RANDOM_DATA_PATH" ]; then
        echo "Source file found. Running filter..."
        python -m data_generation.filter_minigrid_data \
            --input "$RANDOM_DATA_PATH" \
            --output "$FILTERED_DATA_PATH" \
            --min_return "$FILTER_THRESHOLD"

        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            echo "ERROR: Python data filtering script failed with exit code $exit_code."
            exit 1
        fi
        echo "Python filtering script finished. Checking if file exists NOW..."
        sleep 1
        ls -l "$FILTERED_DATA_PATH"
        if [ ! -f "$FILTERED_DATA_PATH" ]; then
             echo "ERROR: Filtered file '$FILTERED_DATA_PATH' STILL does not exist after filtering attempt!"
             exit 1
        else
             echo "File '$FILTERED_DATA_PATH' successfully created."
        fi
    else
        echo "ERROR: Cannot filter, random data file not found: $RANDOM_DATA_PATH"
        exit 1
    fi
else
    echo "Check result: Filtered file FOUND."
    echo "Filtered data file already exists: $FILTERED_DATA_PATH"
fi
echo "--- End Step 2 ---"
sleep 1

echo "[Step 3] Running training..."
echo "Running Decision Transformer on filtered data..."
python -m train.trainer --config experiments/config_dt_minigrid.yaml

echo "Running Decision Transformer + HCAM on filtered data..."
python -m train.trainer --config experiments/config_dt_hcam_minigrid.yaml

echo "--- Script Finished ---"