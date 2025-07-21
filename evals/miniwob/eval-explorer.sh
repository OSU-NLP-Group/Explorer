export CONTROLLER_ADDR=
export OPENAI_API_KEY=


# Total number of tasks
total_tasks=$(cat evals/miniwob/available_tasks.txt | wc -l)

id=1  # Initialize task ID counter

for task in $(cat evals/miniwob/available_tasks.txt)
do
    echo "Task ID: $id, task: $task"
    mkdir -p LOG_DIR/$task
    python -u -m evals.miniwob.main_slm --env $task --llm chatgpt --num-episodes 4 --erci 1 --irci 3 --sgrounding --use-dynamic-seed --output-dir LOG_DIR/$task --add-class-subset --ckpt-path <CKPT_PATH>

    id=$((id+1))  # Increment task ID counter
done

