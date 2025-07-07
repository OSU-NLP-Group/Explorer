import traceback
import logging

from PIL import Image
from .utils import pil_to_b64
from in_domain_eval.utils import call_gpt4v
import tiktoken
from in_domain_eval.prompts.browser_prompt import PARSE_CORRECTNESS_EXAMPLES

class ParseCorrectorAgent:
    def __init__(self, args):
        self.args = args

        self.sm = """You are an expert in parsing arbitrary strings to correct format without adding any new information. Given a malformed string with missing information, your task is to output the python string in correct format with all relevant fields while strictly following the format.

        Rules:
1. If there are multiple dicts in the input string, output information corresponding to the first dict only.
2. Correct the format of the dict but do NOT add any new information.
3. The valid dictionary fields are ['action', 'action_natural_language', 'idx', 'value']. Make sure there is no other field in the output.
4. If the field is missing in the input, output the field with None value.

The output should be in below format:
*OUTPUT FORMAT*: 
{"action": <ACTION>:str, "action_natural_language": <ACTION_IN_NATURAL_LANGUAGE>:str, "idx": <element_idx chosen from the second screen>:int, "value": <the text to enter>:str}
"""
    # @profile
    def act(self, input_str):
        # import pdb; pdb.set_trace()
        try:
            messages = self.create_request(input_str)
            
            ans_1st_pass, _ = call_gpt4v(self.args, messages)


        except Exception as e:
            result = None
            ans_1st_pass = ""
            finish_reason = ""
            usage = {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0}
            logging.info('error in trajectory verifier agent')
            # logging.info(traceback.format_exc())
        
        response = ans_1st_pass
        return response

    def create_request(self, input_str):

        prompt = f"""Input string: {input_str}"""
        
        messages = [{"role":"system","content":[{"type": "text", "text": self.sm}]}]

        examples = PARSE_CORRECTNESS_EXAMPLES

        for (x, y) in examples:
            messages.append(
                {
                    "role": "system",
                    "name": "example_user",
                    "content": [{"type": "text", "text": x}],
                }
            )
            messages.append(
                {
                    "role": "system",
                    "name": "example_assistant",
                    "content": [{"type": "text", "text": y}],
                }
            )
        messages.append({"role":"user","content":[{"type": "text", "text":prompt}]})
        
        return messages