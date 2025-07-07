import random
import traceback
import json
import sys
import requests
import time
import logging
from openai import AsyncAzureOpenAI
from .browser_env import ScriptBrowserEnv, ActionParsingError
# from .processors import ImageObservationProcessor
from .processors import ImageObservationProcessor

from .actions import create_playwright_action, create_id_based_action, create_none_action
from PIL import Image
from .utils import pil_to_b64
from in_domain_eval.prompts.browser_prompt import BROWSER_ONE_STEP_EXAMPLES, BROWSER_DEEP_SEARCH_EXAMPLES, BROWSER_RANDOM_WALKER_EXAMPLES
import ast
from in_domain_eval.parse_corrector_agent import ParseCorrectorAgent

def handle_popups(page):
    # List of common popup selectors to close
    popup_selectors = [
        'text=I agree',              # Common for cookie consent
        'text=No thanks',            # Common for sign-in popups
        'text=Stay signed out',      # Specific for Google's "Stay signed out" popup
        'button.close',              # Common for generic close buttons
        'button[aria-label="Close"]', # Aria label close buttons
        '[data-test="popup-close"]', # Data-test attribute close buttons
        '[aria-label="Dismiss"]'     # Aria label dismiss buttons
    ]
    
    for selector in popup_selectors:
        print(f"Checking for popup close button: {selector}")
        try:
            if page.is_visible(selector):
                print('flag 1')
                page.click(selector)
                print(f"Clicked popup close button: {selector}")
        except Exception as e:
            print(f"Popup close button not found or already handled: {selector} - {e}")

class ScriptBrowserEnvAgent:
    def __init__(self, args, browser_type='chrome', viewport_size={'width': 1280, 'height': 2048}):
        self.args = args
        image_observation_type = 'image_som'
        self.viewport_size = viewport_size
        self.browser_env = ScriptBrowserEnv(args, browser_type=browser_type, viewport_size=viewport_size)

        '''
        print(self.browser_env.page.content())
        # close any popups
        self.browser_env.page.screenshot(path="screenshot_before.png")
        handle_popups(self.browser_env.page)
        self.browser_env.page.screenshot(path="screenshot_after.png")
        '''

        self.image_processor = ImageObservationProcessor(
            args, image_observation_type, viewport_size
        )
        # self.cur_url = None
        self.num_top_res = 10

        self.temperature = 0
        self.max_tokens = 200

        self.sm = """You are an autonomous intelligent agent tasked with navigating a web browser. Given the state of browser environment and the description of the next action in natural language, your task is to map the natural language action to one of the discrete actions that can be performed on the browser.
        The actions you can perform fall into several categories:

Page Operation Actions:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0.
`hover [id]`: Hover over an element with id.
`press [key_comb]`:  Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).
`scroll [direction=down|up]`: Scroll the page up or down.

To be successful, it is very important to follow the following rules:
1. You should only issue an action that is valid given the current observation
2. You should only issue one action at a time.
3. The ID of element should be present in the screenshot. Do not generate any ID which is not visible in the screenshot.
4. There is no need to press the Enter key after typing unless specified.
5. You should follow the examples to reason step by step and then issue the next action.
6. Generate the action in the correct format. Start with a "In summary, the next action I will perform is" phrase, followed by action inside ``````. For example, "In summary, the next action I will perform is ```click [1234]```".
7. the argument to scroll is 'up' or 'down'. For instance, ```scroll [down]``` or ```scroll [up]```.
"""
        self.answer_phrase = "In summary, the next action I will perform is"

        self.correct_action_agent = ParseCorrectorAgent(args)

    def get_state(self):
        # print('inside get_state')
        # print('self.browser_env.page = ', self.browser_env.page)
        # print('self.browser_env.page.client = ', self.browser_env.page.client)

        # som_image_obs, parsed_html_str = self.image_processor.process(self.browser_env.page, self.browser_env.page.client, intent=None)
        som_image_obs, parsed_html_str = self.image_processor.process_new(self.browser_env.page, self.browser_env.page.client, intent=None)
        
        # som_image_obs, parsed_html_str = self.image_processor.process(self.browser_env.page)
        return {'page': self.browser_env.page, 'client': self.browser_env.page.client, 'content_str': parsed_html_str, 'image_obs': som_image_obs}

    def act(self, action, browser_env_state):
        is_action_valid = True

        # execute that action
        # logging.info('Browser agent NL action = {}'.format(action))
        orig_action = action

        try:
            if action.startswith('json'):
                action = action[4:]

            action = action.strip()
            action = action.replace('```', '')
            action = action.replace("{{", "{")
            action = action.replace("}}", "}")

            start = action.find('{')  # Find the first occurrence of '['
            end = action.rfind('}')  # Find the last occurrence of ']'
            
            # Check if '[' and ']' were found and if they are in the right order
            if start != -1 and end != -1 and end > start:
                action = action[start:end + 1]  # Return the longest substring
    
            pred = ast.literal_eval(action)
            assert isinstance(pred, dict)
            assert 'action' in pred
            if pred['action'] == 'type':
                assert 'value' in pred
            if pred['action'] in ['click', 'type']:
                assert 'idx' in pred
        except:
            if self.args.use_gpt_correction:
                try:
                    action = self.correct_action_agent.act(orig_action)
                    
                    pred = ast.literal_eval(action)
                    assert isinstance(pred, dict)
                    assert 'action' in pred
                    if pred['action'] == 'type':
                        assert 'value' in pred
                    if pred['action'] in ['click', 'type']:
                        assert 'idx' in pred
                except:
                    logging.error('Error in parsing the prediction dict {}'.format(action))
                    pred = {'idx': 'regex fail', 'action_natural_language': '', 'action': 'regex fail', 'value': 'regex fail'}
                    logging.error(traceback.format_exc())
                    is_action_valid = False
            else:
                logging.error('Error in parsing the prediction dict {}'.format(action))
                pred = {'idx': 'regex fail', 'action_natural_language': '', 'action': 'regex fail', 'value': 'regex fail'}
                logging.error(traceback.format_exc())
                is_action_valid = False

        action_grounded = pred['action']

        if action_grounded in ['click']:
            idx = pred['idx']
            action_canonical = "{} [{}]".format(action_grounded, idx)
        elif action_grounded in ['type']:
            idx = pred['idx']
            action_canonical = "{} [{}] [{}]".format(action_grounded, idx, pred['value'])
        elif action_grounded in ['scroll']:
            idx = pred['idx']
            action_canonical = "{} [{}]".format(action_grounded, idx)
        elif action_grounded in ['select']:
            action_canonical = f"select [{pred['idx']}] [{pred['value']}]"
        else:
            action_canonical = action_grounded

        try:
            cur_action = create_id_based_action(action_canonical)
        except:
            logging.error('Action parsing error')
            # traceback.print_exc()
            logging.error(traceback.format_exc())

            cur_action = create_none_action()
            is_action_valid = False

        logging.info(f"Action to be executed: {cur_action}")

        self.browser_env.step(cur_action)

        logging.info('URL 0: {}'.format(self.browser_env.page.url))

        try:
            state = self.get_state()
            res = {'content_str': state['content_str'], 'image_obs': state['image_obs']}
        except:
            res = {}

        return pred, res, is_action_valid
    
    def create_request(self, action_nl, browser_env_state, tmp, mt):
        acc_tree = browser_env_state['content_str']
        image_obs = browser_env_state['image_obs']

        # print('action_nl = ', action_nl)

        if self.args.omit_acc_tree:
            prompt = [{"type": "text", "text": f"\nOBJECTIVE: {action_nl}"}, {"type": "image_url", "image_url": {"url": pil_to_b64(Image.fromarray(image_obs))}}]
        else:
            prompt = [{"type": "text", "text": f"OBSERVATION:\n {acc_tree}\nOBJECTIVE: {action_nl}"}, {"type": "image_url", "image_url": {"url": pil_to_b64(Image.fromarray(image_obs))}}]

        messages = [{"role":"system","content":[{"type": "text", "text": self.sm}]}]

        # if self.args.use_web_random_walker:
        examples = BROWSER_RANDOM_WALKER_EXAMPLES
        # elif self.args.use_single_step:
        #     examples = BROWSER_ONE_STEP_EXAMPLES
        # else:
        #     examples = BROWSER_DEEP_SEARCH_EXAMPLES

        for (x, y, z) in examples:
            messages.append(
                {
                    "role": "system",
                    "name": "example_user",
                    "content": [{"type": "text", "text": x}, {"type": "image_url", "image_url": {"url": pil_to_b64(Image.open(z))}}],
                }
            )
            messages.append(
                {
                    "role": "system",
                    "name": "example_assistant",
                    "content": y,
                }
            )
        # print(messages)
        messages.append({"role":"user","content":prompt})
        return messages
        
    
if __name__=='__main__':
    browser_agent = ScriptBrowserEnvAgent()

    # action = 'browser call click on Gmail button'

    # action = 'browser call go to the URL https://www.amazon.com'
    action = 'browser call go to the URL https://en.wikipedia.org/wiki/Apple'

    browser_env_state = browser_agent.get_state()
    img = Image.fromarray(browser_env_state['image_obs'])
    img.save('screenshot_init.png')

    # print('page before = {}'.format(browser_env_state['content_str']))

    browser_env_state = browser_agent.act(action, browser_env_state)
    # print('page after = {}'.format(browser_env_state['content_str']))

    img = Image.fromarray(browser_env_state['image_obs'])
    img.save('screenshot_after_act1.png')

    # second action
    # action = 'browser call Click on the Description link'
    # action = 'type banana into the text box'
    action = 'Click on the Apple Inc link'
    browser_env_state = browser_agent.act(action, browser_env_state)
    # print('page after = {}'.format(browser_env_state['content_str']))

    img = Image.fromarray(browser_env_state['image_obs'])
    img.save('screenshot_after_act2.png')

    print(browser_agent.browser_env.page.url)