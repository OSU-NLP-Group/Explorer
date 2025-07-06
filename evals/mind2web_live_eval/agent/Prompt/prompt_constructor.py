from ..Utils.utils import is_valid_base64
import json5

from .our_prompts import OurPrompts
from jinja2 import Template
from PIL import Image
import base64
from io import BytesIO

import logging

logger = logging.getLogger(__name__)

class BasePromptConstructor:
    def __init__(self):
        pass

def pil_to_b64(img: Image.Image) -> str:
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
        img_b64 = base64.b64encode(byte_data).decode("utf-8")
        img_b64 = "data:image/png;base64," + img_b64
    return img_b64



# Build a prompt for planning based on the our data generation pipeline
class OurPromptConstructor(BasePromptConstructor):
    def __init__(self):
        self.prompt_system = OurPrompts.planning_prompt_system
        self.prompt_user = OurPrompts.planning_prompt_user
        
    def construct(
            self,
            args,
            user_request: str,
            action_history: list,
            acc_tree: str,
            image_obs = None,
            init_url = None,
            step_index = -1
    ) -> list:
        # print('image_obs:', image_obs)

        logger.info('step_index: {}'.format(step_index))

        prompt = self.prompt_user.format(user_request, action_history, acc_tree)

        logger.info('user prompt: {}'.format(prompt))

        messages = [{"role":"system","content": self.prompt_system}]
        
        if args.planning_text_model.startswith('gpt'):
            messages.append({"role":"user","content":
            [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": pil_to_b64(image_obs)}}
            ]})
        else:
            messages.append({"role":"user","content":prompt})

        return messages, image_obs

    # Previous thought, action and reflection are converted to formatted strings
    def stringfy_thought_and_action(self, input_list: list) -> str:
        input_list = json5.loads(input_list, encoding="utf-8")
        str_output = "["
        for idx, i in enumerate(input_list):
            str_output += f'Step{idx + 1}:\"Thought: {i["thought"]}, Action: {i["action"]}, Reflection:{i["reflection"]}\";\n'
        str_output += "]"
        return str_output
