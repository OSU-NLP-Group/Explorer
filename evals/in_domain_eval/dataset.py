import os, json
import torch
from PIL import Image

SYSTEM_MESSAGE = '''You are an expert at completing instructions on Webpage screens. 
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
'''

SYSTEM_MESSAGE_GS = '''You are an expert at completing instructions on Webpage screens. 
               You will be presented with a screenshot image with some numeric tags.
               If you decide to click somewhere, you should choose the numeric element idx that is the closest to the location you want to click.  
               You should decide the action to continue this instruction.
               You will be given the accessibility tree of the current screen in the format: '[element_idx] [role] [alt text or button name]'.
               Here are the available actions:
{"action": "goto", "action_natural_language": str, "value": <the url to go to>}
{"action": "google_search", "action_natural_language": str, "value": <search query for google>}
{"action": "click", "action_natural_language": str, "idx": <element_idx>}
{"action": "type", "action_natural_language": str, "idx": <element_idx>, "value": <the text to enter>}
{"action": "select", "action_natural_language": str, "idx": <element_idx>, "value": <the option to select>}
{"action": "scroll [up]", "action_natural_language": str}
{"action": "scroll [down]", "action_natural_language": str}
Your final answer must be in the above format.
'''

USER_MESSAGE = '''Here is the screenshot image: <|image_1|>\n
The instruction is to {}. 
History actions:
{}\n\n
Here is the screen information:
{}\n\n
Think about what you need to do with current screen, and output the action in the required format in the end. '''

class WebTrajDataCollator:
    def __init__(self, args, processor):
        self.args = args
        self.processor = processor

    def __call__(self, overall_task, acc_tree, som_screenshot_path, action_history=[]):
        
        system_message_prompt = SYSTEM_MESSAGE

        # print(system_message_prompt)

        # system message
        system_message = {
            'role': 'system',
            'content': system_message_prompt,
        }
        
        prompt_message = {
            'role': 'user',
            'content': USER_MESSAGE.format(overall_task, action_history, acc_tree),
        }
        image = Image.open(som_screenshot_path)

        if self.args.model == 'phi-3.5':
            prompt = self.processor.tokenizer.apply_chat_template(
                [system_message, prompt_message], tokenize=False, add_generation_prompt=True
            )
            batch = self.processor(prompt, [image], return_tensors="pt")
        else:
            messages_new = []
            messages_new.append(system_message)
            user_prompt = USER_MESSAGE.format(overall_task, action_history, acc_tree)

            user_prompt_0, user_prompt_1 = user_prompt.split('<|image_1|>')
            messages_new.append({"role": "user", "content": [{"type": "text", "text": user_prompt_0}, {"type": "image"}, {"type": "text", "text": user_prompt_1}]})
        
            prompt = self.processor.apply_chat_template(messages_new, add_generation_prompt=True)
            batch = self.processor(text=prompt, images=[image], padding=True, return_tensors="pt")

        
        batch.to('cuda')

        return batch