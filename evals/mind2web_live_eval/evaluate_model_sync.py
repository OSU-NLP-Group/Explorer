from evals.mind2web_live_eval.agent.Environment.html_env.browser_env import (
    ScriptBrowserEnv,
)
from evals.mind2web_live_eval.evaluate import *
from evals.mind2web_live_eval.agent.Plan import *
from dataclasses import dataclass

import re
import argparse
import logging

# universal tools
from evals.mind2web_live_eval.agent.Utils.utils import *

# evaluate tools
from evals.mind2web_live_eval.evaluate.evaluate_utils import (
    read_config,
    read_file,
    read_json_file,
)
from evals.mind2web_live_eval.evaluate.evaluate_utils_sync import run_task

from evals.mind2web_live_eval.experiment_results import get_evaluate_result
from tqdm import tqdm
import traceback

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    mode: str
    planning_text_model: str
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


def get_task_range(args, task_mode, file, raw_data_index):
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


def run_experiment(args, task_range, experiment_config):
    all_json_models = experiment_config.config["model"]["json_models"]
    is_json_response = experiment_config.config["model"]["json_model_response"]

    llm_planning_text = create_llm_instance(
        args, experiment_config.planning_text_model, is_json_response, all_json_models
    )

    som_model, caption_model_processor = None, None

    for task_index in tqdm(task_range):
        task_uuid = None
        if experiment_config.config["basic"]["task_mode"] == "batch_tasks":
            task = experiment_config.file[task_index]
            task_name, task_uuid, reference_task_length, reference_evaluate_steps = task
            evaluate_steps = reference_evaluate_steps
            log_task_info(
                task_index, task_name, reference_task_length, reference_evaluate_steps
            )
        elif experiment_config.config["basic"]["task_mode"] == "single_task":
            task_name = experiment_config.single_task_name
            reference_task_length = experiment_config.config["steps"][
                "single_task_action_step"
            ]
            # TODO
            evaluate_steps = experiment_config.config["steps"][
                "single_task_action_step"
            ]
            reference_evaluate_steps = None
            logger.info(f"task_name: {task_name}")

        env = ScriptBrowserEnv(
            args,
            browser_type="chrome",
            viewport_size={
                "width": args.viewport_width,
                "height": args.viewport_height,
            },
        )

        try:
            run_task(
                args=args,
                llm_planning_text=llm_planning_text,
                mode=experiment_config.mode,
                task_mode=experiment_config.config["basic"]["task_mode"],
                task_name=task_name,
                task_uuid=task_uuid,
                config=experiment_config.config,
                write_result_file_path=experiment_config.write_result_file_path,
                reference_task_length=reference_task_length,
                evaluate_steps=evaluate_steps,
                reference_evaluate_steps=reference_evaluate_steps,
                env=env,
                planning_text_model=experiment_config.planning_text_model,
                ground_truth_mode=experiment_config.ground_truth_mode,
                ground_truth_data=experiment_config.ground_truth_data,
                interaction_mode=experiment_config.config["steps"]["interaction_mode"],
                task_index=task_index,
                record_time=experiment_config.record_time,
                som_model=som_model,
                caption_model_processor=caption_model_processor,
            )

            env.close()

        except:
            logger.info("Error in running task...")
            logger.info(traceback.format_exc())
            env.close()

        del env

        if args.debug:
            break

    try:
        get_evaluate_result(args, experiment_config.config["files"]["out_file_path"])
    except:
        logger.info("Error in getting evaluate result...")
        logger.info(traceback.format_exc())

    logger.info("\033[31mAll tasks finished!\033[0m")
    logger.info("\033[31mPress Enter to exit...\033[0m")


def main_sync(
    args,
    planning_text_model="gpt-4-turbo",
    single_task_name="",
    raw_data_index=-1,
    observation_mode="dom",
    ground_truth_mode=False,
    toml_path=None,
):
    config = read_config(args.toml_path)
    validate_config(config, observation_mode, planning_text_model)

    file = None
    if config["basic"]["task_mode"] == "batch_tasks":
        file = read_file(file_path=config["files"]["batch_tasks_file_path"])
        task_range = get_task_range(
            args, config["basic"]["task_mode"], file, raw_data_index
        )
    elif config["basic"]["task_mode"] == "single_task":
        task_range = get_task_range(
            args,
            config["basic"]["task_mode"],
            config["files"]["batch_tasks_file_path"],
            -1,
        )

    print(f"task_range: {task_range}")

    record_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    write_result_file_path = generate_result_file_path(config)
    ground_truth_data = load_ground_truth_data(config, ground_truth_mode)

    experiment_config = ExperimentConfig(
        mode=observation_mode,
        planning_text_model=planning_text_model,
        ground_truth_mode=ground_truth_mode,
        single_task_name=single_task_name,
        config=config,
        ground_truth_data=ground_truth_data,
        write_result_file_path=write_result_file_path,
        record_time=record_time,
        file=file,
    )

    run_experiment(args, task_range, experiment_config)
