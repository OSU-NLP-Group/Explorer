from evals.mind2web_live_eval.evaluate import *
from evals.mind2web_live_eval.agent.Plan import *
from playwright.async_api import Page
from evals.mind2web_live_eval.agent.Environment.html_env.async_env import (
    ActionExecutionError,
)
import re
import toml
import json
import traceback
import os
from evals.mind2web_live_eval.agent.Environment import (
    ActionExecutionError,
    create_action,
)
from evals.mind2web_live_eval.agent.Plan import Planning
from evals.mind2web_live_eval.agent.Utils.utils import save_screenshot, is_valid_base64
from evals.mind2web_live_eval.evaluate import (
    FinishTaskEvaluator,
    TaskLengthEvaluator,
    URLEvaluator,
    ElementEvaluator,
)
from evals.mind2web_live_eval.logs import logger
from evals.mind2web_live_eval.agent.Environment.html_env.actions_sync import (
    create_id_based_action,
    create_none_action,
)


def adjust_max_action_step(conditions, current_info, encountered_errors, increase_step):
    total_increase = 0
    for condition_type, keywords in conditions.items():
        for keyword in keywords:
            if (
                keyword in current_info[condition_type]
                and keyword not in encountered_errors
            ):
                print(
                    f"Detected '{keyword}' in {current_info[condition_type]}, suggesting increase by {increase_step} steps."
                )
                total_increase += increase_step
                encountered_errors.add(keyword)
    return total_increase, encountered_errors


def get_netloc(url: str) -> str:
    """Extract the domain name, for example, extract 'zhihu' from 'zhihu.com', extract 'google' from 'www.google.com.hk'"""
    url = urlparse(url)
    try:
        if url.netloc.startswith("www"):
            netloc = re.findall(".*?\.(.*?)\..*?", url.netloc)[0]
        else:
            netloc = re.findall("(.*?)\..*?", url.netloc)[0]
    except:
        netloc = ""
    return netloc


def step_evaluate(
    page: Page, evaluate_steps=[], input_path=None, element_value=None, args=None
):
    """Evaluate step score"""
    step_score = 0
    match_result = []
    for evaluate in evaluate_steps:
        if evaluate["score"] != 1:
            match_function = evaluate["match_function"]
            if match_function == "url_exactly_match":
                score = URLEvaluator.url_exact_match(
                    page.url, evaluate["reference_answer"], evaluate["key"]
                )
            elif match_function == "url_included_match":
                score = URLEvaluator.url_include_match(
                    page.url, evaluate["reference_answer"], evaluate["key"]
                )
            elif match_function == "url_semantic_match":
                score = URLEvaluator.url_semantic_match_sync(
                    page.url, evaluate["reference_answer"], evaluate["key"], args
                )
                # print(score, "url_semantic_match")
            elif match_function == "element_path_exactly_match":
                input_netloc = get_netloc(page.url)
                method = evaluate["method"]
                score = ElementEvaluator.path_exact_match(
                    input_path,
                    evaluate["reference_answer"],
                    method,
                    page.content(),
                    input_netloc,
                    evaluate["netloc"],
                )
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
                        path_score = ElementEvaluator.path_exact_match(
                            input_path,
                            evaluate["path"],
                            "selector",
                            page.content(),
                            input_netloc,
                            evaluate["netloc"],
                        )
                        if path_score == 0:
                            # print("Path mismatch in value evaluation")
                            score = 0
                        else:
                            score = ElementEvaluator.element_value_exact_match(
                                element_value,
                                evaluate["reference_answer"],
                                input_netloc,
                                evaluate["netloc"],
                            )
                    else:
                        score = ElementEvaluator.element_value_exact_match(
                            element_value,
                            evaluate["reference_answer"],
                            input_netloc,
                            evaluate["netloc"],
                        )
                    # print(score, "element_value_exactly_match",
                    #       element_value, "*", evaluate["reference_answer"])
                else:
                    score = 0
            elif match_function == "element_value_included_match":
                if input_path is not None and element_value is not None:
                    input_netloc = get_netloc(page.url)
                    if "path" in evaluate.keys():
                        path_score = ElementEvaluator.path_exact_match(
                            input_path,
                            evaluate["path"],
                            "selector",
                            page.content(),
                            input_netloc,
                            evaluate["netloc"],
                        )
                        if path_score == 0:
                            # print("Path mismatch in value evaluation")
                            score = 0
                        else:
                            score = ElementEvaluator.element_value_include_match(
                                element_value,
                                evaluate["reference_answer"],
                                input_netloc,
                                evaluate["netloc"],
                            )
                    else:
                        score = ElementEvaluator.element_value_include_match(
                            element_value,
                            evaluate["reference_answer"],
                            input_netloc,
                            evaluate["netloc"],
                        )
                    # print(score, "element_value_included_match",
                    #       element_value, "*", evaluate["reference_answer"])
                else:
                    score = 0
            elif match_function == "element_value_semantic_match":
                # logger.info('inside element_value_semantic_match')

                # logger.info('input_path = {}'.format(input_path))
                # logger.info('element_value = {}'.format(element_value))

                if input_path is not None and element_value is not None:
                    input_netloc = get_netloc(page.url)

                    if len(element_value) > 0:
                        # logger.info('inside element_value_semantic_match len(element_value) > 0')
                        if "path" in evaluate.keys():
                            path_score = ElementEvaluator.path_exact_match(
                                input_path,
                                evaluate["path"],
                                "selector",
                                page.content(),
                                input_netloc,
                                evaluate["netloc"],
                            )
                            if path_score == 0:
                                # logger.info('inside element_value_semantic_match path_score == 0')
                                # print("Path mismatch in value evaluation")
                                score = 0
                            else:
                                # logger.info('inside element_value_semantic_match path_score != 0')
                                score = (
                                    ElementEvaluator.element_value_semantic_match_sync(
                                        element_value,
                                        evaluate["reference_answer"],
                                        input_netloc,
                                        evaluate["netloc"],
                                        args,
                                    )
                                )
                        else:
                            # logger.info('inside element_value_semantic_match len(element_value) == 0')
                            score = ElementEvaluator.element_value_semantic_match_sync(
                                element_value,
                                evaluate["reference_answer"],
                                input_netloc,
                                evaluate["netloc"],
                                args,
                            )
                        # print(score, "element_value_semantic_match",
                        #       element_value, "*", evaluate["reference_answer"])
                else:
                    # logger.info('inside element_value_semantic_match input_path is None or element_value is None')
                    score = 0
            elif match_function == "text_exact_match":
                pass  # TODO
            elif match_function == "text_include_match":
                pass
            elif match_function == "text_semantic_match":
                pass

            # logger.info('evaluate["score"] = {}'.format(evaluate["score"]))
            # logger.info('score = {}'.format(score))
            # logger.info('type(evaluate["score"]) = {}'.format(type(evaluate["score"])))
            # logger.info('type(score) = {}'.format(type(score)))

            evaluate["score"] = max(evaluate["score"], score)
        if evaluate["score"] >= 1:
            match_result.append(
                {evaluate["match_function"]: evaluate["reference_answer"]}
            )
        step_score += evaluate["score"]
    # print("current step score:", step_score, "/", len(evaluate_steps))
    # print("current step match result:", match_result)
    return evaluate_steps, match_result
    # print(evaluate_steps)


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
        toml_path = "configs/setting.toml"

    with open(toml_path, "r") as f:
        config = toml.load(f)

    return config


def run_task(
    args,
    llm_planning_text,
    mode,
    task_mode,
    task_name,
    task_uuid,
    config,
    write_result_file_path,
    reference_task_length,
    evaluate_steps,
    reference_evaluate_steps,
    env,
    planning_text_model,
    ground_truth_mode,
    ground_truth_data,
    interaction_mode,
    task_index,
    record_time=None,
    som_model=None,
    caption_model_processor=None,
):
    print("inside run_task")

    env.setup("about:blank")

    response_error_count = 0
    response_total_count = 0

    # Related to the HTML environment
    observation = ""
    observation_VforD = ""
    error_description = ""
    previous_trace = []

    # Related to response
    out_put = None

    # If all are matched, the task is completed
    task_finished = False
    task_global_status = ""
    human_interaction_stop_status = False

    # Configuration related to controlling the length of steps
    conditions = config["conditions"]
    increase_step = config["steps"]["batch_tasks_condition_step_increase"]
    encountered_errors = set()

    try:
        current_info = {"URL": env.page.url}
    except:
        current_info = {"URL": "about:blank"}

    num_steps = 0
    num_global_steps = 0  # avoid infinite loop in case of planning errors

    if task_mode == "single_task":
        max_steps = int(reference_task_length)
    elif task_mode == "batch_tasks":
        max_steps = int(
            max(
                config["steps"]["batch_tasks_max_action_step"],
                1.5 * reference_task_length,
            )
        )
    additional_steps = 0

    # Store the results of the planning process for a task
    task_result = {}
    task_result["task_name"] = task_name
    task_result["id"] = task_uuid
    task_result["reference_task_length"] = reference_task_length
    steps_list = []

    init_url = "about:blank"
    env.setup(init_url)

    assert mode == "ours"

    num_tries = 0

    for i in range(3):
        try:
            observation = env.get_obs(
                image_path=None,
                som_model=som_model,
                caption_model_processor=caption_model_processor,
            )
            break
        except:
            logger.info(
                "Error in get_obs() try {}: {}".format(
                    num_tries, traceback.format_exc()
                )
            )
            num_tries += 1
            time.sleep(5)

    if num_tries == 3:
        raise Exception("Failed to load initial obs. after 3 tries")

    # print('observation = {}'.format(observation))

    logger.info(f"Acc tree: {observation['content_str']}")

    if args.output_screenshots:
        os.makedirs(
            os.path.join(config["files"]["out_file_path"], str(task_index)),
            exist_ok=True,
        )
        observation["image_obs"].save(
            os.path.join(
                config["files"]["out_file_path"],
                str(task_index),
                "som_screenshot_{}.png".format(num_global_steps),
            )
        )

    all_json_models = config["model"]["json_models"]
    is_json_response = config["model"]["json_model_response"]

    # llm_planning_text = create_llm_instance(
    #         args, planning_text_model, is_json_response, all_json_models)

    while (
        num_steps < max_steps + additional_steps
        and num_global_steps < args.num_global_steps
    ):
        error_message = ""
        total_step_score = 0
        step_reward = {}
        status_description = ""

        logger.info(f"Global step: {num_global_steps}")

        print(f"Global step: {num_global_steps}")

        logger.info("**🤖 The agent is in the process of starting planning 🤖**")

        try:
            # n_tries = 0
            # while n_tries < args.n_tries:
            response_total_count += 1
            # n_tries += 1
            # logger.info(f"n_tries: {n_tries}")

            try:
                out_put = Planning.plan_sync(
                    task_index=task_index,
                    step_index=num_global_steps,
                    args=args,
                    llm_planning_text=llm_planning_text,
                    config=config,
                    user_request=task_name,
                    text_model_name=planning_text_model,
                    previous_trace=previous_trace,
                    observation=observation,
                    feedback=error_description,
                    mode=mode,
                    observation_VforD=observation_VforD,
                    status_description=status_description,
                )

                logger.info("output = {}".format(out_put))

                # if out_put is not None:
                # break
            except Exception as e:
                out_put = None
                response_error_count += 1
                logger.info(traceback.format_exc())
                # continue

            if out_put:
                each_step_dict = {}
                each_step_dict["step_index"] = num_steps
                each_step_dict["dict_result"] = out_put
                # execute_action, current_trace, path, element_value = parse_current_trace(
                # out_put, env, step_reward)

                action_grounded, pred = out_put

                try:
                    cur_action = create_id_based_action(
                        action_grounded, args.use_google_api
                    )
                except:
                    logger.error("Action parsing error")
                    # traceback.print_exc()
                    logger.error(traceback.format_exc())

                    cur_action = create_none_action()
                    is_action_valid = False

                if "idx" in pred:
                    element_id = pred["idx"]
                else:
                    element_id = None

                element_value = pred["value"]

                tree = observation["tree"]

                # logger.info(f"tree.uniqueId2nodeId = {tree.uniqueId2nodeId}")

                if element_id in tree.uniqueId2nodeId:
                    node_id = tree.uniqueId2nodeId[element_id]
                    selector = tree.get_selector(node_id)
                else:
                    selector = "none"  # default

                current_trace = {
                    "thought": "",
                    "action": pred["action_natural_language"],
                    "reflection": "",
                }

                each_step_dict["current_trace"] = current_trace
                each_step_dict["selector"] = selector
                # each_step_dict["execute_action"] = execute_action
                each_step_dict["element_value"] = element_value

                logger.info(f"-- Planning output: {out_put}")
                logger.info(f"-- Current trace: {current_trace}")
                # logger.info(f"-- Action: {execute_action}")
                logger.info(f"-- Selector: {selector}")
                logger.info(f"-- Element value: {element_value}")

                logger.info(
                    "**🤖 The agent is in the process of starting evaluation 🤖**"
                )
                if task_mode == "batch_tasks":
                    evaluate_steps, match_result = step_evaluate(
                        page=env.page,
                        evaluate_steps=evaluate_steps,
                        input_path=selector,
                        element_value=element_value,
                        args=args,
                    )
                    for evaluate in evaluate_steps:
                        total_step_score += evaluate["score"]

                    each_step_dict["score"] = (
                        str(total_step_score)
                        + " / "
                        + str(len(reference_evaluate_steps))
                    )
                    each_step_dict["match_func_result"] = match_result

                    logger.info(
                        f"-- Current evaluation score: {total_step_score} / {len(reference_evaluate_steps)}"
                    )
                    logger.info(f"-- Current evaluate match result: {match_result}")

                    # get status of the task with global reward
                    if step_reward:
                        each_step_dict["step_reward"] = step_reward
                        task_global_status = step_reward.get("status")
                    else:
                        each_step_dict["step_reward"] = {}

                    if total_step_score == len(reference_evaluate_steps):
                        # steps_list.append(each_step_dict)
                        task_finished = True
                        # break

                logger.info(
                    "**🤖 The agent is in the process of executing the action 🤖**"
                )

                try:
                    env.step(cur_action)
                    previous_trace.append(current_trace)
                    error_description = ""
                    logger.info("-- Successfully execute the action ")
                except ActionExecutionError as ee:
                    error_message = ee.message
                    logger.info("-- Failed to execute the action")
                    logger.error(f"ActionExecutionError occurred: {error_message}")
                    error_description = error_message

                try:
                    # if a new tab is opened by clicking, switch to the new tab
                    logging.info("no of tabs = {}".format(len(env.context.pages)))

                    if len(env.context.pages) > 1:
                        env.page = env.context.pages[-1]
                        logging.info(env.page)
                        env.page.bring_to_front()
                        env.page.client = env.page.context.new_cdp_session(env.page)  # type: ignore[attr-defined]

                        logging.info(env.page)

                    image_path = None

                    observation = env.get_obs(
                        image_path=image_path,
                        som_model=som_model,
                        caption_model_processor=caption_model_processor,
                    )
                    logger.info(f"Acc tree: {observation['content_str']}")
                except:
                    logger.info("Error in get_obs() {}".format(traceback.format_exc()))
                    num_global_steps += 1
                    continue

                if args.output_screenshots:
                    os.makedirs(
                        os.path.join(config["files"]["out_file_path"], str(task_index)),
                        exist_ok=True,
                    )
                    observation["image_obs"].save(
                        os.path.join(
                            config["files"]["out_file_path"],
                            str(task_index),
                            "som_screenshot_{}.png".format(num_global_steps + 1),
                        )
                    )

                # URL after executing the action
                each_step_dict["step_url"] = env.page.url
                each_step_dict["step_url"] = env.page.url
                each_step_dict["error_message"] = error_message
                each_step_dict["previous_trace"] = str(previous_trace)

                logger.info(f"-- The URL is: {env.page.url}")

                current_info = {"URL": env.page.url}
                logger.info(
                    f"**🤖 Time Step: {num_steps + 1}, Total steps: {max_steps + additional_steps} 🤖**"
                )
                step_increase, encountered_errors = adjust_max_action_step(
                    conditions, current_info, encountered_errors, increase_step
                )
                additional_steps += step_increase
                num_steps += 1
                steps_list.append(each_step_dict)
                if num_steps >= 25 or task_global_status == "finished" or task_finished:
                    logger.info("task_global_status = {}".format(task_global_status))
                    break

            if interaction_mode:
                logger.info(
                    "Press Enter to proceed to the next action, or type 'q' to quit the task. If you encounter any unexpected issues such as network connection errors or captcha challenges, please resolve them manually now."
                )
                a = input()
                if a.lower() == "q":
                    logger.info("User requested to quit the program.")
                    human_interaction_stop_status = True
                    break
        except:
            logger.info(f"Error in step {num_global_steps}")
            logger.info(traceback.format_exc())

        num_global_steps += 1

    # ! 3. Task evaluation and scoring
    if task_mode == "batch_tasks":
        # step score
        total_step_score = 0
        for evaluate in evaluate_steps:
            total_step_score += evaluate["score"]
        logger.info(
            f"Total step score: {total_step_score} / {len(reference_evaluate_steps)}"
        )

        # length score
        task_evaluator = TaskLengthEvaluator()
        task_length_score = task_evaluator.task_length_score(
            reference_task_length, num_steps
        )

        logger.info(f"Task length score: {task_length_score}")
        logger.info(
            f"Response error rate: {response_error_count / response_total_count}"
        )

        # finish score
        finish_task_score = FinishTaskEvaluator.finish_task_score(
            len(reference_evaluate_steps), total_step_score
        )
        logger.info(f"Finish task score: {finish_task_score}")

        # Save the status of the task
        if task_finished:
            task_result["status"] = "finished"
        elif task_global_status == "finished":
            task_result["status"] = "llm_finished"
        elif human_interaction_stop_status:
            task_result["status"] = "early_stop"
        else:
            task_result["status"] = "step_limit"

        task_result["LLM_error_rate"] = str(response_error_count / response_total_count)
        task_result["step_list"] = steps_list
        task_result["evaluate_steps"] = reference_evaluate_steps

        json_result_folder = write_result_file_path
        if not os.path.exists(json_result_folder):
            os.makedirs(json_result_folder)
        json_out_file_path = os.path.join(
            json_result_folder, str(task_index) + "_" + str(task_result["id"]) + ".json"
        )
        logger.info(f"Write results to json file: {json_out_file_path}")
        with open(json_out_file_path, "w") as json_file:
            json.dump(task_result, json_file)
