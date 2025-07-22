from evals.mind2web_live_eval.agent.Environment.html_env.async_env import (
    AsyncHTMLEnvironment,
)
from evals.mind2web_live_eval.agent.Environment.html_env.browser_env import (
    ScriptBrowserEnv,
)
from evals.mind2web_live_eval.evaluate import *
from evals.mind2web_live_eval.agent.Plan import *
from dataclasses import dataclass

from evals.mind2web_live_eval.evaluate_model_sync import main_sync

import re
import asyncio
import argparse
import logging

# universal tools
from evals.mind2web_live_eval.agent.Utils.utils import *

# evaluate tools
from evals.mind2web_live_eval.evaluate.evaluate_utils import read_json_file
from evals.mind2web_live_eval.experiment_results import get_evaluate_result
from tqdm import tqdm
import traceback

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    mode: str
    planning_text_model: str
    global_reward_text_model: str
    ground_truth_mode: bool
    single_task_name: str
    config: dict
    ground_truth_data: dict
    write_result_file_path: str
    record_time: str
    file: list


def validate_config(config, observation_mode, observation_model):
    task_mode = config["basic"]["task_mode"]
    batch_tasks_file_path = config["files"]["batch_tasks_file_path"]
    json_model_response = config["model"]["json_model_response"]
    all_json_models = config["model"]["json_models"]
    interaction_mode = config["steps"]["interaction_mode"]

    if observation_mode not in ["dom", "ours"]:
        logger.error(
            "observation mode is not correctly defined! Currently we only support DOM and ours observation."
        )
        exit()

    if interaction_mode not in [True, False]:
        logger.error(
            "interaction_mode is not defined! Try to define whether you want to evaluate the agent in an interactive manner."
        )
        exit()

    if json_model_response and (observation_model not in all_json_models):
        logger.error("Model does not support JSON mode!")
        exit()

    if task_mode == "batch_tasks" and not os.path.exists(batch_tasks_file_path):
        logger.error("batch_tasks_file_path not exist!")
        exit()


def get_task_range(task_mode, file, raw_data_index):
    print("raw_data_index: ", raw_data_index)
    print("task_mode: ", task_mode)
    print("len(file): ", len(file))

    if task_mode == "batch_tasks":
        if raw_data_index != -1:
            re_result = re.split(r"\s|,", str(raw_data_index))
            raw_data_start_index = int(re_result[0])
            raw_data_end_index = int(re_result[-1]) + 1

            return range(raw_data_start_index, raw_data_end_index)
        else:
            if len(args.task_list) > 0:
                return args.task_list
            else:
                raw_data_start_index = 0
                raw_data_end_index = len(file)

                return range(raw_data_start_index, raw_data_end_index)

    elif task_mode == "single_task":
        # return range(0, 1)
        return range(4, 5)
    else:
        logger.error("task_mode error!")
        exit()


def log_task_info(
    task_index, task_name, reference_task_length, reference_evaluate_steps
):
    logger.info("*" * 100)
    logger.info(f"task index: {task_index}")
    logger.info(f"task name: {task_name}")
    logger.info(f"task reference length: {reference_task_length}")
    logger.info(f"raw data annotation: {reference_evaluate_steps}")


def generate_result_file_path(config):
    return os.path.join(config["files"]["out_file_path"], "json_result")


def load_ground_truth_data(config, ground_truth_mode):
    if ground_truth_mode:
        ground_truth_file_path = config["files"]["ground_truth_file_path"]
        if not os.path.exists(ground_truth_file_path):
            logger.error("ground_truth_file_path not exist!")
            exit()
        return read_json_file(ground_truth_file_path)
    return None


def create_html_environment(args, mode, headless, viewport_width, viewport_height):
    return AsyncHTMLEnvironment(
        args,
        mode=mode,
        max_page_length=8192,
        headless=headless,
        slow_mo=1000,
        current_viewport_only=False,
        viewport_size={"width": viewport_width, "height": viewport_height},
        save_trace_enabled=False,
        sleep_after_execution=0.0,
        locale="en-US",
        use_vimium_effect=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the web agent in different modes."
    )

    parser.add_argument("--index", type=int, default=-1)
    parser.add_argument(
        "--single_task_name",
        type=str,
        default="Find Dota 2 game and add all DLC to cart in steam.",
    )
    parser.add_argument(
        "--planning_text_model",
        type=str,
        choices=[
            "gpt-3.5-turbo",
            "gpt4o_2",
            "gpt-4o",
            "gpt-4o-mini",
            "phi-3v",
            "phi-3.5v",
            "phi3mini",
            "Mistral-7B-Instruct",
            "llava-v1.6-mistral-7b",
            "llava-v1.6-vicuna-13b-hf",
            "qwen2-vl-7b",
            "qwen2-vl-72b",
            "intern-vl-8b",
            "phi3mini",
        ],
        default="gpt-3.5-turbo",
    )
    parser.add_argument("--viewport-width", type=int, default=1280)
    parser.add_argument("--viewport-height", type=int, default=720)

    parser.add_argument("--toml-path", type=str, default="configs/setting.toml")
    parser.add_argument("--log-dir", type=str, default="./")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--use-flash-attention", action="store_true", help="Use Flash Attention"
    )
    parser.add_argument("--bf16", action="store_true", help="Use BF16")
    parser.add_argument(
        "--omit-verbose-logging",
        action="store_true",
        help="Omit verbose logging in planning module",
    )
    parser.add_argument(
        "--n-tries", type=int, default=3, help="no. of tries if not in json format"
    )
    parser.add_argument("--ckpt-path", type=str, default="", help="path to Phi-3V ckpt")
    parser.add_argument(
        "--add-repetition-penalty", action="store_true", help="add rep penalty for SLM"
    )
    parser.add_argument(
        "--repetition-penalty", type=float, default=1.1, help="rep penalty for SLM"
    )
    parser.add_argument(
        "--temp", type=float, help="decoding temperature", required=True
    )

    parser.add_argument(
        "--add-first-action",
        action="store_true",
        help="auto add the first goto url action",
    )
    parser.add_argument(
        "--skip-image-step0",
        action="store_true",
        help="skip (blank) image at step 0 of VLM, modify prompt to skip reference of image at step 0",
    )
    parser.add_argument("--omit-tab-name", action="store_true", help="")

    parser.add_argument(
        "--num-global-steps", type=int, default=25, help="no. of global steps"
    )
    parser.add_argument(
        "--press-enter-after-fill",
        action="store_true",
        help="press enter after filling the form",
    )
    parser.add_argument(
        "--use-visible-acc-tree", action="store_true", help="use visible acc tree"
    )
    parser.add_argument("--debug", action="store_true", help="just run a single task")
    parser.add_argument(
        "--output-screenshots", action="store_true", help="", default=True
    )

    parser.add_argument(
        "--omit-image", action="store_true", help="use only textual input"
    )
    parser.add_argument(
        "--use-complete-acc-tree",
        action="store_true",
        help="use complete acc tree for observation",
    )
    parser.add_argument("--use-greedy", action="store_true", help="use greedy decoding")
    parser.add_argument(
        "--num-crops", type=int, default=16, help="no. of crops for Phi-3.5V model"
    )
    parser.add_argument(
        "--max-len-acctree", type=int, default=None, help="max length for acc tree"
    )
    parser.add_argument(
        "--screenshot-height",
        type=int,
        default=-1,
        help="height of screenshot to crop to if > -1",
    )
    parser.add_argument(
        "--process-sleep", type=int, default=5, help="sleep time in process_new"
    )
    parser.add_argument(
        "--use-google-api",
        action="store_true",
        help="use google api directly for google search",
    )

    # Define argument for a list of numbers with the --numbers flag
    parser.add_argument(
        "--task-list",
        type=int,
        nargs="+",
        default=[],
        help="list of tasks to evaluate on",
    )

    args = parser.parse_args()

    print(args)

    setup_logging(args.log_dir)

    logging.info(args)

    main_sync(
        args,
        planning_text_model=args.planning_text_model,
        single_task_name=args.single_task_name,
        raw_data_index=args.index,
        observation_mode="ours",
    )
