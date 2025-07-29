import traceback
import logging

from PIL import Image
from .utils import pil_to_b64
from traj_gen.llm_utils import call_gpt4v
from traj_gen.utils import calc_num_tokens
import tiktoken


class CaptchaDetectionAgent:
    def __init__(self, args):
        self.args = args

        self.sm = """You are an expert in evaluating whether the given webpage screenshot contains a captcha or not. Given the last snapshot of the web page, your goal is to decide whether the webpage contains a captcha or not.
Output "Yes" if the given webpage shows a captcha, otherwise "No". 
*IMPORTANT*
Format your response into a line as shown below:

Answer: "Yes" or "No"
"""

    def act(self, image_obs):
        try:
            messages = self.create_request(image_obs)

            ans_1st_pass, _ = call_gpt4v(self.args, messages)

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
            logging.info("error in captcha agent")
            logging.info(traceback.format_exc())

        response = ans_1st_pass
        return response

    def create_request(self, image_obs):
        prompt = f"""The screenshot of the web page is shown in the image."""

        messages = [{"role": "system", "content": [{"type": "text", "text": self.sm}]}]
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(Image.open(image_obs))},
                    },
                ],
            }
        )

        return messages
