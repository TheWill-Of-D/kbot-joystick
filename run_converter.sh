#!/bin/bash

# --- Configuration ---
if [ -z "$1" ]; then
  echo "Usage: $0 <run_number>"
  echo "Example: $0 18"
  exit 1
fi
RUN_NUMBER=$1

# Define key project directories
PROJECT_ROOT="/workspace/kbot-joystick"
BASE_PROJECT_DIR="${PROJECT_ROOT}/humanoid_walking_task"
CHECKPOINTS_PARENT_DIR="${BASE_PROJECT_DIR}"
OUTPUT_ASSETS_DIR="assets"

# --- Find Checkpoint (Original Logic) ---

RUN_DIR_NAME="run_${RUN_NUMBER}"
CHECKPOINT_DIR="${CHECKPOINTS_PARENT_DIR}/${RUN_DIR_NAME}/checkpoints"

if [ ! -d "$CHECKPOINT_DIR" ]; then
  echo "Error: Checkpoint directory '$CHECKPOINT_DIR' not found."
  exit 1
fi

LATEST_CHECKPOINT_PATH=$(ls -t "${CHECKPOINT_DIR}"/ckpt.*.bin 2>/dev/null | head -n 1)

if [ -z "$LATEST_CHECKPOINT_PATH" ]; then
  echo "Error: No 'ckpt.*.bin' files found in '$CHECKPOINT_DIR'."
  exit 1
fi

echo "Found latest checkpoint: $LATEST_CHECKPOINT_PATH"

# --- MODIFICATION START: Use the Correct Training Code ---

# 1. Define the path to the training code for this specific run.
RUN_DIR_PATH="${CHECKPOINTS_PARENT_DIR}/${RUN_DIR_NAME}"
TRAINING_CODE_PATH="${RUN_DIR_PATH}/training_code.py"

echo "Searching for training code at: $TRAINING_CODE_PATH"

# 2. Check that the training code file actually exists.
if [ ! -f "$TRAINING_CODE_PATH" ]; then
    echo "Error: Could not find training_code.py for run ${RUN_NUMBER}."
    echo "Please ensure it is saved at the location above."
    exit 1
fi

# 3. Define the destination 'train.py' that convert.py expects to import.
#    We will temporarily replace this file with the correct version for the run.
DEST_TRAIN_PY="${PROJECT_ROOT}/train.py"

echo "Temporarily using training code from run ${RUN_NUMBER} for conversion..."
cp "$TRAINING_CODE_PATH" "$DEST_TRAIN_PY"

if [ $? -ne 0 ]; then
    echo "Error: Failed to copy training code to ${DEST_TRAIN_PY}."
    exit 1
fi

# --- MODIFICATION END ---


# --- Run Conversion (Original Logic) ---

# Extract the step value
CHECKPOINT_BASENAME=$(basename "$LATEST_CHECKPOINT_PATH")
STEP_AND_EXT=${CHECKPOINT_BASENAME#ckpt.}
STEP_VALUE=${STEP_AND_EXT%.bin}
echo "Extracted step value: $STEP_VALUE"

# Construct the output model name and path
MODEL_NAME="e0_${RUN_DIR_NAME}_${STEP_VALUE}"
OUTPUT_KINFER_FILENAME="${MODEL_NAME}.kinfer"
OUTPUT_KINFER_PATH="${OUTPUT_ASSETS_DIR}/${OUTPUT_KINFER_FILENAME}"
mkdir -p "$OUTPUT_ASSETS_DIR"

# Form and run the command
echo "Running conversion:"
python -m convert_n1 "$LATEST_CHECKPOINT_PATH" "$OUTPUT_KINFER_PATH"

# Capture the exit code of the python script
CONVERT_EXIT_CODE=$?


# --- NEW: Cleanup Step ---
# It's good practice to restore your workspace to its original state.
# This command will restore train.py from your last git commit.
# If you don't use git, you could create a backup and restore it instead.
echo "Restoring original train.py..."
git checkout -- "$DEST_TRAIN_PY"
# ---


# Check if the conversion failed and exit if it did
if [ $CONVERT_EXIT_CODE -ne 0 ]; then
    echo "Conversion script failed with exit code $CONVERT_EXIT_CODE."
    exit 1
fi

echo "Conversion complete. Output at: $OUTPUT_KINFER_PATH"