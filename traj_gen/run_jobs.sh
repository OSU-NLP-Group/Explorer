#!/bin/bash

# Number of concurrent processes
max_concurrent=64

# CSV file containing URLs

csv_file="/path/to/url/csv/file"
data_dir="DATA_DIR"


# Number of total jobs
n_total_jobs=$(awk -F, 'NR>1 {print $1}' "$csv_file" | wc -l)

# Function to count running jobs with the specific name
count_running_jobs() {
  pgrep -f "python -m traj_gen.main" | wc -l
}

# Function to run a single process
run_process() {
  local id=$1
  local url=$2
  echo "Starting process $id with URL: $url"
  
  local start_time=$(date +%s)

  # Assign a unique DISPLAY number based on the process ID
  local display_num=$((1000 + $id))  # Use DISPLAY numbers starting from :1000
  local display=":$display_num"

  # Start an Xvfb server on the specified display
  Xvfb $display -screen 0 1920x1280x16 &  # Start Xvfb in the background
  local xvfb_pid=$!

  export DISPLAY=$display

  # Run the Python script with the URL from the CSV file
  timeout 500 python -m traj_gen.main --model-dir $data_dir/$id --init-url "$url" --max-steps 10 --max-global-steps 20 --viewport-width 1280 --viewport-height 1080

  exit_code=$?

  if [ $exit_code -eq 0 ]; then
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo "Process $id finished successfully in $duration seconds"
  elif [ $exit_code -eq 124 ]; then
    echo "Process $id timed out"
  else
    echo "Process $id failed with exit code $exit_code"
  fi

  # Stop the Xvfb server
  kill $xvfb_pid
}

# Loop through each URL in the CSV file
i=1
awk -F, 'NR>1 {print $1}' "$csv_file" | while read -r url; do
  run_process $i "$url" &

  # Get the current number of background processes
  while [ $(count_running_jobs) -ge $max_concurrent ]; do
    # Wait for any background process to finish
    sleep 1
  done

  i=$((i + 1))
done

# Wait for all background processes to finish
wait

echo "All processes have finished."

