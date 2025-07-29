import traceback
import logging

from PIL import Image
from .utils import pil_to_b64
from .actions import create_id_based_action, create_none_action
from traj_gen.llm_utils import call_gpt4v
import re
from traj_gen.utils import calc_num_tokens
import tiktoken


class TaskProposalAgent:
    def __init__(self, args, browser_env, image_processor):
        self.args = args
        self.browser_env = browser_env
        self.image_processor = image_processor

        self.sm = f"""What does this webpage show? Imagine you are real user on this webpage. Given the webpage screenshot and parsed HTML/accessibility tree, please provide a single task that a user might perform on this page and the corresponding first action towards completing that task.
    Do the following step by step:
    1. Generate a single task that a user might perform on this webpage. Be creative and come up with diverse tasks
    2. Given the webpage screenshot and parsed HTML/accessibility tree, generate the first action towards completing that task (in natural language form).
    3. Given the webpage screenshot, parsed HTML/accessibility tree, and the natural language action, generate the grounded version of that action.

    *ACTION SPACE*: Your action space is: [`click [element ID]`, `type [element ID] [content]`, `select [element ID] [content of option to select]`, `scroll [up]`, `scroll [down]`, and `stop`].
    Action output should follow the syntax as given below:
    `click [element ID]`: This action clicks on an element with a specific id on the webpage.
    `type [element ID] [content]`: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing. Both the content and the id should be within square braces as per the syntax. 
    `select [element ID] [content of option to select]`: Select an option from a dropdown menu. The content of the option to select should be within square braces. When you get (select and option) tags from the accessibility tree , you need to select the serial number (element_id) corresponding to the select tag , not the option, and select the most likely content corresponding to the option as input.
    `scroll [down]`: Scroll the page down. 
    `scroll [up]`: Scroll the page up.

    *IMPORTANT* To be successful, it is important to STRICTLY follow the below rules:
    
    *  Action generation rules *
    1. You should generate a single atomic action at each step.
    2. The action should be an atomic action from the given vocabulary - click, type, scroll (up or down) or stop
    3. The arguments to each action should be within square braces. For example, "click [127]", "type [43] [content to type]", "scroll [up]", "scroll [down]".
    4. The natural language form of action (corresponding to the field "action_in_natural_language") should be consistent with the grounded version of the action (corresponding to the field "grounded_action"). Do NOT add any additional information in the grounded action. For example, if a particular element ID is specified in the grounded action, a description of that element must be present in the natural language action. 
    5. If the type action is selected, the natural language form of action ("action_in_natural_language") should always specify the actual text to be typed. 
    6. You should issue a “stop” action if the current webpage asks to login or for credit card information. 
    7. To input text, there is NO need to click textbox first, directly type content. After typing, the system automatically hits the `ENTER` key.
    8. STRICTLY Avoid repeating the same action (click/type) if the webpage remains unchanged. You may have selected the wrong web element.
    9. Do NOT use quotation marks in the action generation.

    *  Task proposal rules *
    1. You should propose tasks that are relevant to the website and can be completed using the website.
    2. You should only propose tasks that do not require login to execute the task.
    3. You should propose tasks that are clear and specific.
    4. For each task, provide concrete information or constraints, and use mock-up information (identifier, number, personal information, name, attributes, etc.) to make the task more specific and realistic.
    5. The task description should provide all the necessary information to complete the task.
    6. The task should be feasible to complete by a real user and should not require any additional information that is not available on the website.

    The output should be in below format:
    *OUTPUT FORMAT*: Please give a short analysis of the screenshot, parsed HTML/accessibility tree, then put your answer within ``` ```, for example, "In summary, the proposed task and the corresponding action is: ```{{"task": <TASK>:str, "action_in_natural_language":<ACTION_IN_NATURAL_LANGUAGE>:str, "grounded_action": <ACTION>:str}}```"

    """

    def act(self, acc_tree, image_obs):
        is_action_valid = True

        try:
            messages = self.create_request(acc_tree, image_obs)

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
            is_action_valid = False

        response = ans_1st_pass
        logging.info(f"response = {response}")

        try:
            import ast

            try:
                pred = re.findall(r"```(.*?)```", response, re.DOTALL)[-1]
            except:
                try:
                    pred = response.split("```")[-2]  # handle case with three ```
                except:
                    logging.error("Error in parsing the prediction ``````")
                    pred = ""
                    logging.error(traceback.format_exc())
                    is_action_valid = False

            try:
                if pred.startswith("json"):
                    pred = pred[4:]
                pred = ast.literal_eval(pred)
            except:
                logging.error("Error in parsing the prediction dict {}".format(pred))
                pred = {
                    "task": "regex fail",
                    "action_in_natural_language": "regex fail",
                    "grounded_action": "regex fail",
                }
                logging.error(traceback.format_exc())
                is_action_valid = False

            action_grounded = pred["grounded_action"]

            try:
                cur_action = create_id_based_action(action_grounded)
            except:
                logging.error("Action parsing error")
                logging.error(traceback.format_exc())

                cur_action = create_none_action()
                is_action_valid = False

            logging.info(f"Action to be executed: {cur_action}")

            is_success = self.browser_env.step(cur_action)
            if not is_success:
                is_action_valid = False

            logging.info("URL 0: {}".format(self.browser_env.page.url))

        except:
            pred = {
                "task": "regex fail",
                "action_in_natural_language": "regex fail",
                "grounded_action": "regex fail",
            }
            browser_env_state = {}
            logging.error(traceback.format_exc())
            is_action_valid = False

        return response, pred, is_action_valid

    def create_request(self, acc_tree, image_obs):
        prompt = [
            {
                "type": "text",
                "text": f"WEBSITE URL: {self.args.init_url}\n PARSED HTML/ACCESSIBILITY TREE:\n {acc_tree}",
            },
            {
                "type": "image_url",
                "image_url": {"url": pil_to_b64(Image.fromarray(image_obs))},
            },
        ]

        messages = [{"role": "system", "content": [{"type": "text", "text": self.sm}]}]

        messages.append({"role": "user", "content": prompt})
        return messages
