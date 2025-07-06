from evals.mind2web_live_eval.evaluate import *
from evals.mind2web_live_eval.agent.Plan import *
from playwright.async_api import Page
from evals.mind2web_live_eval.agent.Environment.html_env.async_env import AsyncHTMLEnvironment, ActionExecutionError

import re
import toml
import json
import traceback
import os
from evals.mind2web_live_eval.agent.Environment import ActionExecutionError, create_action
from evals.mind2web_live_eval.agent.Plan import Planning
from evals.mind2web_live_eval.agent.Utils.utils import save_screenshot, is_valid_base64
from evals.mind2web_live_eval.evaluate import FinishTaskEvaluator, TaskLengthEvaluator, URLEvaluator, ElementEvaluator
from evals.mind2web_live_eval.logs import logger


def read_file(file_path="./data/example/example_130.json"):
    """Read labeled data"""
    return_list = []
    with open(file_path, encoding='utf-8') as f:
        test_data = json5.load(f)
    for task in test_data:
        task_name = task["task"]
        evaluation_data = task["evaluation"]
        reference_task_length = task["reference_task_length"]
        task_name_id = task["index"]
        reference_evaluate_steps = []
        for i, evaluation in enumerate(evaluation_data):
            match_function = evaluation["match_function_name"]
            if "url" in match_function:
                try:
                    key = evaluation["content"]["key"]
                    reference_answer = evaluation["content"]["reference_answer"]
                    reference_evaluate_steps.append({"match_function": match_function,
                                                     "key": key, "reference_answer": reference_answer, "score": 0})
                except:
                    logger.error(
                        f"url error in task {task_name_id}, step {i}, match_function: {match_function}")
                    exit(1)
            elif "element_path" in match_function:
                try:
                    reference_answer = evaluation["content"]["reference_answer"]
                    method = evaluation["method"]
                    netloc = evaluation["content"]["netloc"]
                    reference_evaluate_steps.append({"match_function": match_function, "method": method,
                                                     "reference_answer": reference_answer, "netloc": netloc,
                                                     "score": 0})
                except:
                    logger.error(
                        f"element_path error in task {task_name_id}, step {i}, match_function: {match_function}")
                    exit(1)
            elif "element_value" in match_function:
                try:
                    reference_answer = evaluation["content"]["reference_answer"]
                    netloc = evaluation["content"]["netloc"]
                    if "path" in evaluation["content"].keys():
                        path = evaluation["content"]["path"]
                        reference_evaluate_steps.append({"match_function": match_function,
                                                         "reference_answer": reference_answer, "netloc": netloc,
                                                         "path": path, "score": 0})
                    else:
                        reference_evaluate_steps.append({"match_function": match_function,
                                                         "reference_answer": reference_answer, "netloc": netloc,
                                                         "score": 0})
                except:
                    logger.error(
                        f"element_value error in task {task_name_id}, step {i}, match_function: {match_function}")
                    exit(1)
        return_list.append(
            [task_name, task_name_id, reference_task_length, reference_evaluate_steps])

    return return_list


async def adjust_max_action_step(conditions, current_info, encountered_errors, increase_step):
    total_increase = 0
    for condition_type, keywords in conditions.items():
        for keyword in keywords:
            if keyword in current_info[condition_type] and keyword not in encountered_errors:
                print(
                    f"Detected '{keyword}' in {current_info[condition_type]}, suggesting increase by {increase_step} steps.")
                total_increase += increase_step
                encountered_errors.add(keyword)
    return total_increase, encountered_errors


def get_netloc(url: str) -> str:
    """Extract the domain name, for example, extract 'zhihu' from 'zhihu.com', extract 'google' from 'www.google.com.hk' """
    url = urlparse(url)
    try:
        if url.netloc.startswith("www"):
            netloc = re.findall(".*?\.(.*?)\..*?", url.netloc)[0]
        else:
            netloc = re.findall("(.*?)\..*?", url.netloc)[0]
    except:
        netloc = ""
    return netloc


async def step_evaluate(page: Page, evaluate_steps=[], input_path=None, element_value=None, args=None):
    """Evaluate step score"""
    step_score = 0
    match_result = []
    for evaluate in evaluate_steps:
        if evaluate["score"] != 1:
            match_function = evaluate["match_function"]
            if match_function == "url_exactly_match":
                score = URLEvaluator.url_exact_match(
                    page.url, evaluate["reference_answer"], evaluate["key"])
            elif match_function == "url_included_match":
                score = URLEvaluator.url_include_match(
                    page.url, evaluate["reference_answer"], evaluate["key"])
            elif match_function == "url_semantic_match":
                score = await URLEvaluator.url_semantic_match(
                    page.url, evaluate["reference_answer"], evaluate["key"], args)
                # print(score, "url_semantic_match")
            elif match_function == "element_path_exactly_match":
                input_netloc = get_netloc(page.url)
                method = evaluate["method"]
                score = ElementEvaluator.path_exact_match(
                    input_path, evaluate["reference_answer"], method, await page.content(), input_netloc,
                    evaluate["netloc"])
                # print(score, "path_exact_match:", input_path,
                #       "***", evaluate["reference_answer"])
            elif match_function == "element_path_included_match":
                pass
                # * Temporarily not doing

            elif match_function == "element_value_exactly_match":
                if input_path is not None and element_value is not None:
                    input_netloc = get_netloc(page.url)

                    # print(element_value)
                    # print(await page.locator(input_path).input_value())
                    if "path" in evaluate.keys():
                        path_score = ElementEvaluator.path_exact_match(input_path, evaluate["path"], "selector",
                                                                       await page.content(), input_netloc,
                                                                       evaluate["netloc"])
                        if path_score == 0:
                            # print("Path mismatch in value evaluation")
                            score = 0
                        else:
                            score = ElementEvaluator.element_value_exact_match(
                                element_value, evaluate["reference_answer"], input_netloc, evaluate["netloc"])
                    else:
                        score = ElementEvaluator.element_value_exact_match(
                            element_value, evaluate["reference_answer"], input_netloc, evaluate["netloc"])
                    # print(score, "element_value_exactly_match",
                    #       element_value, "*", evaluate["reference_answer"])
                else:
                    score = 0
            elif match_function == "element_value_included_match":
                if input_path is not None and element_value is not None:
                    input_netloc = get_netloc(page.url)
                    if "path" in evaluate.keys():
                        path_score = ElementEvaluator.path_exact_match(input_path, evaluate["path"], "selector",
                                                                       await page.content(), input_netloc,
                                                                       evaluate["netloc"])
                        if path_score == 0:
                            # print("Path mismatch in value evaluation")
                            score = 0
                        else:
                            score = ElementEvaluator.element_value_include_match(
                                element_value, evaluate["reference_answer"], input_netloc, evaluate["netloc"])
                    else:
                        score = ElementEvaluator.element_value_include_match(
                            element_value, evaluate["reference_answer"], input_netloc, evaluate["netloc"])
                    # print(score, "element_value_included_match",
                    #       element_value, "*", evaluate["reference_answer"])
                else:
                    score = 0
            elif match_function == "element_value_semantic_match":
                if input_path is not None and element_value is not None:
                    input_netloc = get_netloc(page.url)

                    if len(element_value) > 0:
                        if "path" in evaluate.keys():
                            path_score = ElementEvaluator.path_exact_match(input_path, evaluate["path"], "selector",
                                                                           await page.content(), input_netloc,
                                                                           evaluate["netloc"])
                            if path_score == 0:
                                # print("Path mismatch in value evaluation")
                                score = 0
                            else:
                                score = await ElementEvaluator.element_value_semantic_match(
                                    element_value, evaluate["reference_answer"], input_netloc, evaluate["netloc"], args)
                        else:
                            score = await ElementEvaluator.element_value_semantic_match(
                                element_value, evaluate["reference_answer"], input_netloc, evaluate["netloc"], args)
                        # print(score, "element_value_semantic_match",
                        #       element_value, "*", evaluate["reference_answer"])
                else:
                    score = 0
            elif match_function == "text_exact_match":
                pass  # TODO
            elif match_function == "text_include_match":
                pass
            elif match_function == "text_semantic_match":
                pass

            evaluate["score"] = max(evaluate["score"], score)
        if evaluate["score"] >= 1:
            match_result.append(
                {evaluate["match_function"]: evaluate["reference_answer"]})
        step_score += evaluate["score"]
    # print("current step score:", step_score, "/", len(evaluate_steps))
    # print("current step match result:", match_result)
    return evaluate_steps, match_result
    # print(evaluate_steps)


def parse_current_trace(response: dict, env: AsyncHTMLEnvironment, step_reward: dict):
    logger.info(f"Response: {response}")
    
    thought = response["description"].get("thought")
    action_type = response['action_type']
    action_input = response['value']
    action = response["description"].get("action")
    reflection = step_reward.get(
        "description") if step_reward else ""
    current_trace = {"thought": thought,
                     "action": action, "reflection": reflection}
    element_value = ""
    selector = None

    try:
        element_id = int(response['id'])
    except:
        element_id = 0
    
    if action_type == "google_search":
        element_value = action_input
    elif action_type == "goto":
        element_value = action_input
    elif action_type in ["fill_form", "fill_search", "click", "select_option"]:
        try:
            selector = env.tree.get_selector_and_xpath(
                env.tree.nodeDict[element_id])
            element_value = env.tree.get_element_value(
                env.tree.nodeDict[element_id])
            if action_type in ["fill_form", "fill_search"]:
                element_value = action_input
        except:
            logger.info(
                "Failed to obtain element_id from the accessibility tree.")
            element_id = 0
            action_type = "None"
    else:
        selector = None
        element_id = 0
    execute_action = create_action(
        elementid=element_id, action_type=action_type, action_input=action_input)
    return execute_action, current_trace, selector, element_value


def read_config(toml_path=None):
    """
    Reads a TOML configuration file from the given path or the default path
    and returns its content as a dictionary.

    Args:
        toml_path (str, optional): The path to the TOML configuration file.
                                           If None, use the default path.

    Returns:
        dict: The content of the configuration file.
    """
    if toml_path is None:
        # default_path = os.path.join(os.path.dirname(__file__), 'default_settings.toml')
        toml_path = 'configs/setting.toml'

    with open(toml_path, 'r') as f:
        config = toml.load(f)

    return config


