import traceback
import logging

from PIL import Image
from .utils import pil_to_b64

# from in_domain_eval.utils import call_gpt4v
from web_traj_gen.action_prediction_agent import call_gpt4v

import tiktoken


class GptTaskCompletionAgent:
    def __init__(self, args):
        self.args = args

        self.sm = """You are an expert at completing instructions on Webpage screens. 
               You will be presented with a screenshot image with some numeric tags.
               If you decide to click somewhere, you should choose the numeric idx that is the closest to the location you want to click.  
               You should decide the action to continue this instruction.

               Here are all possible actions:
{"action": "click", "action_natural_language": str, "idx": <element_idx chosen from the second screen>}
{"action": "hover", "action_natural_language": str, "idx": <element_idx chosen from the second screen>}
{"action": "enter", "action_natural_language": str, "idx": <element_idx chosen from the second screen>}
{"action": "type", "action_natural_language": str, "idx": <element_idx chosen from the second screen>, "value": <the text to enter>}
{"action": "select", "action_natural_language": str, "idx": <element_idx chosen from the second screen>, "value": <the option to select>}

*  Action generation rules *
1. You should generate a single action (in dictionary format) at each step.
2. The action should be an atomic action from the given vocabulary - click, type, hover, enter, and select.
3. Stricly follow the format of the action as mentioned above. Do NOT generate anything other than the dictionary with the above keys.

The output should be in below format:
{"action": <ACTION>:str, "action_natural_language": <ACTION_IN_NATURAL_LANGUAGE>:str, "idx": <element_idx chosen from the second screen>:int}
"""
        self.user_message = """The instruction is to {}. 
History actions:
{}\n\n
Here is the screen information:
{}\n\n
Think about what you need to do with current screen, and output the action in the required format in the end.\n
Here is the screenshot image: """

    def act(self, overall_task, acc_tree, som_screenshot_path, action_history=[]):
        try:
            messages = self.create_request(
                overall_task, acc_tree, som_screenshot_path, action_history
            )

            ans_1st_pass, _ = call_gpt4v(self.args, messages)

        except Exception as e:
            result = None
            ans_1st_pass = ""
            finish_reason = ""
            usage = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
            logging.info("error in trajectory verifier agent")
            # logging.info(traceback.format_exc())

        response = ans_1st_pass
        return response

    def create_request(
        self, overall_task, acc_tree, som_screenshot_path, action_history
    ):
        prompt = self.user_message.format(overall_task, action_history, acc_tree)

        messages = [{"role": "system", "content": [{"type": "text", "text": self.sm}]}]
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": pil_to_b64(Image.open(som_screenshot_path))
                        },
                    },
                ],
            }
        )

        # logging.info(messages)

        return messages
