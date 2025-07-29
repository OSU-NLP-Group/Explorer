import os
import random
import traceback
from .task_proposal_agent import TaskProposalAgent
from .task_refiner_agent import TaskRefinerAgent
from .task_summarization_flow import TaskSummarizationAgent
import json
from .trajectory_verifier import TrajectoryVerifierAgent
from .captcha_detection_agent import CaptchaDetectionAgent
from PIL import Image
import argparse
import logging
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from .browser_env import ScriptBrowserEnv
from .processors import ImageObservationProcessor
import re

logger = logging.getLogger("__main__")


class Explorer:
    def __init__(self, args):
        self.args = args

        self.viewport_size = {
            "width": args.viewport_width,
            "height": args.viewport_height,
        }
        self.image_observation_type = "image_som"

        self.browser_env = ScriptBrowserEnv(
            args, browser_type="chrome", viewport_size=self.viewport_size
        )

        self.init_setup_error = False
        try:
            self.browser_env.setup(args.init_url)
        except:
            self.init_setup_error = True
            logging.info("Error in setting up the environment. Exiting...")
            logging.info(traceback.format_exc())
            return
        self.image_processor = ImageObservationProcessor(
            args, self.image_observation_type, self.viewport_size
        )

        self.task_proposal_agent = TaskProposalAgent(
            args, self.browser_env, self.image_processor
        )
        self.task_refiner_agent = TaskRefinerAgent(
            args, self.browser_env, self.image_processor
        )
        self.summarization_agent = TaskSummarizationAgent(
            args, self.browser_env, self.image_processor
        )
        self.verifier_agent = TrajectoryVerifierAgent(args)

        self.captcha_detection_agent = CaptchaDetectionAgent(args)

    def get_state(self):
        som_image_obs, parsed_html_str = self.image_processor.process_new(
            self.browser_env.page,
            self.browser_env.page.client,
            use_id_selector=True,
            intent=None,
        )

        html = self.browser_env.page.content()

        return {
            "page": self.browser_env.page,
            "client": self.browser_env.page.client,
            "content_str": parsed_html_str,
            "image_obs": som_image_obs,
            "html": html,
        }

    def run(self, ex_log_dir="."):
        if self.init_setup_error:
            return [], "Error in setting up the environment", False

        task_trajectory_data = {}
        task_trajectory_data["init_url"] = self.args.init_url
        task_trajectory_data["viewport-width"] = self.args.viewport_width
        task_trajectory_data["viewport-height"] = self.args.viewport_height

        task_trajectory_data["actions"] = []
        completed = False

        task_refinement_history = []
        action_history = []
        step = 0
        execution_id = 0

        try:
            while step < self.args.max_steps and execution_id <= 2:
                action = {}
                logging.info(f"Step {step}:\n")
                if completed:
                    break

                # get state of the environment
                if self.browser_env.page is not None:
                    try:
                        browser_env_state = self.get_state()
                    except:
                        logging.info(
                            "Error in getting state, resetting the environment..."
                        )
                        traceback.print_exc()
                        logging.info(traceback.format_exc())
                        # reset the environment
                        self.browser_env.setup(self.args.init_url)

                        task_trajectory_data["actions"] = []
                        task_refinement_history = []
                        action_history = []
                        step = 0
                        execution_id += 1
                        continue

                    if self.args.print_parsed_tree:
                        logging.info(
                            "acc_tree = {}".format(browser_env_state["content_str"])
                        )

                    action["acc_tree_before"] = browser_env_state["content_str"]
                    # action['html_before'] = browser_env_state['html']
                    with open(
                        os.path.join(ex_log_dir, f"html_{step}.html"),
                        "w",
                        encoding="utf-8",
                    ) as f1:
                        f1.write(browser_env_state["html"])

                    if not self.args.no_dump_screenshots:
                        self.browser_env.page.screenshot(
                            path=os.path.join(ex_log_dir, f"screenshot_{step}.png")
                        )

                        img = Image.fromarray(browser_env_state["image_obs"])
                        img.save(os.path.join(ex_log_dir, f"screenshot_som_{step}.png"))
                else:
                    browser_env_state = None

                # check if current page contains a captcha
                if step == 0:
                    captcha_response = self.captcha_detection_agent.act(
                        os.path.join(ex_log_dir, f"screenshot_{step}.png")
                    )
                    logging.info("captcha_response = {}".format(captcha_response))

                    is_captcha = captcha_response.split("Answer:")[-1].strip().lower()

                    if is_captcha == "yes":
                        logging.info("Captcha detected. Terminating the traj.")
                        return [], "Captcha detected", False

                if step == 0:
                    response, pred, is_action_valid = self.task_proposal_agent.act(
                        browser_env_state["content_str"], browser_env_state["image_obs"]
                    )
                else:
                    response, pred, is_action_valid = self.task_refiner_agent.act(
                        browser_env_state["content_str"],
                        browser_env_state["image_obs"],
                        action_history,
                        refined_goal,
                    )

                logging.info(f"pred = {pred}")

                new_action_nl, new_action_grounded, refined_goal = (
                    pred["action_in_natural_language"],
                    pred["grounded_action"],
                    pred["task"],
                )

                # get element id from new_action_grounded
                try:
                    match = re.search(r"\[(\d+)\]", new_action_grounded)
                    element_id = match.group(1)

                    # get bbox coordinates from som_id_info
                    som_id_info = self.image_processor.som_id_info
                    bounding_box_coord = {
                        "x": som_id_info[element_id][0],
                        "y": som_id_info[element_id][1],
                        "width": som_id_info[element_id][2],
                        "height": som_id_info[element_id][3],
                    }
                except:  # scroll action
                    bounding_box_coord = None

                logging.info("Agent response: {}".format(response))

                logging.info("Action (NL): {}\n".format(new_action_nl))
                logging.info("Action (grounded): {}\n".format(new_action_grounded))

                logging.info(f"refined_goal: {refined_goal}\n")

                action["step_action_nl"] = new_action_nl
                action["new_action_grounded"] = new_action_grounded
                action["bounding_box_coord"] = bounding_box_coord
                action["step_refined_goal"] = refined_goal
                action["step_reasoning_response"] = response

                task_refinement_history.append(refined_goal)
                action_history.append(new_action_nl)

                # ground / execute the action
                if new_action_grounded == "stop":
                    completed = True
                    break

                logging.info("URL: {}".format(self.browser_env.page.url))

                if is_action_valid:
                    action["URL_after"] = self.browser_env.page.url
                    task_trajectory_data["actions"].append(action)

                logging.info("##############################\n\n")
                step += 1
        except:
            logging.info("Error in step {}".format(step))

            # put traceback in logging log
            logging.error("{}".format(traceback.format_exc()))
            step += 1

        # summarize the task description using history
        screenshot_history = [
            os.path.join(ex_log_dir, f"screenshot_som_{i}.png") for i in range(step + 1)
        ]
        summarization_response, summarization_pred = self.summarization_agent.act(
            action_history, screenshot_history
        )

        # verify the trajectory
        user_intent = summarization_pred

        history = [
            action["step_action_nl"] for action in task_trajectory_data["actions"]
        ]
        img_path = os.path.join(ex_log_dir, "screenshot_final.png")

        logging.info("user_intent = {}".format(user_intent))
        logging.info("history = {}".format(history))

        self.browser_env.page.screenshot(path=img_path)

        self.browser_env.close()

        try:
            response.raise_for_status()

            last_page_md = response.content.decode("utf-8")
        except:
            last_page_md = None

        if self.args.use_all_screenshots_verifier:
            screenshot_history = [
                os.path.join(ex_log_dir, f"screenshot_{i}.png") for i in range(step + 1)
            ] + [img_path]
            verifier_agent_response = self.verifier_agent.act(
                user_intent, history, screenshot_history, last_page_md
            )
        else:
            verifier_agent_response = self.verifier_agent.act(
                user_intent, history, img_path, last_page_md
            )

        logging.info("verifier_agent_response = {}".format(verifier_agent_response))

        task_trajectory_data["task_summary"] = user_intent
        task_trajectory_data["verifier_agent_response"] = verifier_agent_response

        return task_trajectory_data, verifier_agent_response, True


def to_raw_string(s):
    return s.replace("\\", "\\\\")


def setup_logging(ex_log_dir):
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create a new file handler
    log_file = os.path.join(ex_log_dir, "step_simulator_flow.log")
    logging.basicConfig(
        level=logging.INFO,
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(message)s",
    )


def main(args):
    # set seed
    random.seed(args.seed)

    # create a default unique model dir if not specified
    if args.model_dir is None:
        args.model_dir = "model_" + str(random.randint(0, 1000000))

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir, exist_ok=True)

    flow = Explorer(args)

    setup_logging(args.model_dir)

    task_trajectory_data, verifier_agent_response, is_traj_success = flow.run(
        args.model_dir
    )

    if not is_traj_success:
        return

    # dump the task trajectory data
    with open(os.path.join(args.model_dir, "task_trajectory_data.json"), "w") as f:
        json.dump(task_trajectory_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-steps", type=int, default=5, help="Maximum number of steps to simulate"
    )
    parser.add_argument(
        "--print-parsed-tree",
        action="store_true",
        help="Print the parsed tree in stdout",
    )
    parser.add_argument(
        "--no-dump-screenshots",
        action="store_true",
        help="Do NOT dump screenshots of each step in screenshots/",
    )
    parser.add_argument(
        "--model-dir", type=str, default=None, help="Directory to save the models"
    )
    parser.add_argument("--seed", type=int, default=736537, help="Random seed")
    parser.add_argument(
        "--init-url",
        type=str,
        default="https://www.amazon.com/",
        help="initial url for the browser env",
    )
    parser.add_argument(
        "--temp-refiner",
        type=float,
        default=0.01,
        help="temperature for the refiner agent",
    )
    parser.add_argument(
        "--omit-acc-tree", action="store_true", help="omit the accessibility tree"
    )
    parser.add_argument(
        "--viewport-width", type=int, default=1280, help="viewport width"
    )
    parser.add_argument(
        "--viewport-height", type=int, default=720, help="viewport height"
    )
    parser.add_argument(
        "--print-num-toks",
        action="store_true",
        help="print the token count for each module",
        default=False,
    )
    parser.add_argument(
        "--deployment",
        type=str,
        choices=["gpt-4o", "gpt-4o-mini"],
        default="gpt-4o",
        help="API model deployment",
    )
    parser.add_argument(
        "--use-all-screenshots-verifier",
        action="store_true",
        help="use all screenshots for verifier",
        default=True,
    )
    parser.add_argument(
        "--temp-summ-verf",
        type=float,
        default=0.01,
        help="temperature for the summarizer and verifier agents",
    )

    args = parser.parse_args()
    print(args)

    main(args)
