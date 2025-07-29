import traceback
import logging

from PIL import Image
from .utils import pil_to_b64
from traj_gen.llm_utils import call_gpt4v
import re
import os
from traj_gen.utils import calc_num_tokens
import tiktoken


class TaskSummarizationAgent:
    def __init__(self, args, browser_env, image_processor):
        self.args = args

        self.sm = """Given a list of actions performed on the website {website_url} and the corresponding screenshots ACTION_LIST: [{action_list}] Your task is to come up with a single task description that will be accomplished by performing these actions in the given sequence on the website. 

*IMPORTANT*
0. The task must contain some actions: "Buy, Book, Find, Check, Choose, show me, search, browse, get, compare, view, give me, add to cart, ...', ideally involving transactions/finding information on a specific product or service.
1. You should propose tasks that are clear and specific
2. The task description should provide all the necessary information to complete the task.
3. The task description must indicate the domain of website at the end of the task with format: "... on task_website", for instance, "Purchase a laptop on amazon", "Book a hair appointment on yelp", etc.
4. The task should be feasible to complete by a real user and should not require any additional information that is not specified in this input.
5. The task description should specify constraints like given budget, product features, and other specifications that can narrow down the search to a particular item/product
6. Do NOT use any quotation marks (either single or double) in the task description.

The output should be in below format:
*OUTPUT FORMAT*: Please first give some analysis of the actions and screenshots and then output the overall task description. put your answer within ``` ```, for example, "In summary, the answer is: ```<TASK_DESCRIPTION>:str```".

"""
        self.sm_abstract = """Given a web based task description, suggest a way to modify it to make it more abstract. You should mimic the thought process of a human user to come up with an intent that a real user has in mind before attempting the task."""

    def act(self, action_list, screenshot_list):
        try:
            messages = self.create_request(action_list, screenshot_list)

            ans_1st_pass, _ = call_gpt4v(
                self.args, messages, temperature=self.args.temp_refiner
            )

            if self.args.print_num_toks:
                n_inp_tokens = calc_num_tokens(messages)
                encoding = tiktoken.encoding_for_model("gpt-4o")
                n_op_tokens = len(encoding.encode(ans_1st_pass))
                n_tokens = n_inp_tokens + n_op_tokens

                logging.info(f"Number of tokens: {n_tokens}")

        except Exception as e:
            result = None
            ans_1st_pass = ""
            finish_reason = ""
            usage = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
            logging.info(traceback.format_exc())

        summary_response_1 = ans_1st_pass
        logging.info(f"summarization response = {summary_response_1}")

        try:
            import ast

            match = re.search(r"```(.*?)```", summary_response_1, re.DOTALL)
            result = match.group(1).strip()

            # properly escape quotation marks
            # result = result.replace('"', '\\"').replace("'", "\\'")
            pred_1 = result
        except:
            pred_1 = "regex fail"
            logging.info(traceback.format_exc())

        logging.info(f"pred = {pred_1}")

        return summary_response_1, pred_1

    def create_request(self, action_list, screenshot_list):
        prompt = []

        for screenshot in screenshot_list:
            if os.path.exists(screenshot):
                prompt += [
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(Image.open(screenshot))},
                    }
                ]

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": self.sm.format(
                            action_list=action_list, website_url=self.args.init_url
                        ),
                    }
                ],
            }
        ]

        messages.append({"role": "user", "content": prompt})
        return messages
