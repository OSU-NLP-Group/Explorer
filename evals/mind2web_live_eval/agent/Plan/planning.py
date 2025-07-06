from ..Utils.utils import print_info, print_limited_json
from evals.mind2web_live_eval.agent.Prompt import *
from evals.mind2web_live_eval.agent.LLM import *
from .action import *
import time
import json5
from .action import ResponseError
from evals.mind2web_live_eval.logs import logger
import ast
import io
from PIL import Image
from urllib.parse import urlparse
import re

def is_valid_url(url):
    try:
        result = urlparse(url)
        # Check if the URL has both a valid scheme (e.g., http, https) and netloc (domain)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def extract_first_url(text):
    # Regular expression pattern to match URLs
    url_pattern = r'(https?://[^\s]+)'
    
    # Search for the first occurrence of the pattern
    match = re.search(url_pattern, text)
    
    # Return the matched URL if found, else return None
    return match.group(0) if match else None

class InteractionMode:
    def __init__(self, text_model=None, visual_model=None):
        self.text_model = text_model
        self.visual_model = visual_model

    def execute(self, status_description, user_request, previous_trace, observation, feedback, observation_VforD):
        pass


class Planning:

   
    @staticmethod
    def plan_sync(
        task_index,
        step_index,
        args,
        llm_planning_text,
        config,
        user_request,
        text_model_name,
        previous_trace,
        observation,
        feedback,
        mode,
        observation_VforD,
        status_description,
    ):
        acc_tree = observation['content_str']

        if args.max_len_acctree:
            acc_tree = acc_tree[:args.max_len_acctree]

        image_obs_som = observation['image_obs']

        if args.screenshot_height > 0:
            image_obs_som = image_obs_som.crop((0, 0, image_obs_som.width, args.screenshot_height))

        if args.omit_image:
            image_obs_som = None

        action_history = [x['action'] for x in previous_trace]
        logger.info(f"action_history: {action_history}")
        
        planning_request, image_som = OurPromptConstructor().construct(
            args, user_request, action_history, acc_tree, image_obs_som, step_index=step_index)
        
        if not args.omit_verbose_logging:
            logger.info(
                f"\033[32mDOM_based_planning_request:\n{planning_request}\033[0m\n")
        
        # logger.info(f"planning_text_model: {self.text_model.model}")
        if args.skip_image_step0 and step_index < 2: # skip reference to image in step 0 prompt
            planning_response = llm_planning_text.request_sync(planning_request)
        else:
            planning_response = llm_planning_text.request_sync(planning_request, image_som)

        if not args.omit_verbose_logging:
            logger.info(f"\033[34mPlanning_Response:\n{planning_response}\033[0m")

        # parse the response into fields
        is_action_valid = True

        try:
            logging.info(f'planning_response: {planning_response}')

            if planning_response.startswith('json'):
                planning_response = planning_response[4:]
            
            if '|end|' in planning_response:
                planning_response = planning_response.replace('|end|', '')

            action = planning_response.strip()
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

            if pred['action'] in ['type', 'google_search', 'goto', 'select']:
                if pred['action'] == 'goto' and 'value' not in pred:
                    pred['value'] = pred['idx']
                
                if pred['action'] == 'goto' and not is_valid_url(pred['value']):
                    parsed_url = extract_first_url(pred['action_natural_language'])
                    if parsed_url:
                        logger.info(f"Extracted URL: {parsed_url}")
                        pred['value'] = parsed_url
                        
                assert 'value' in pred
            else:
                pred['value'] = ''
            if pred['action'] in ['click', 'type']:
                assert 'idx' in pred
            
            if pred['action'] == 'type':
                grounded_action = f"type [{pred['idx']}] [{pred['value']}]"
            elif pred['action'] == 'click':
                grounded_action = f"click [{pred['idx']}]"
            elif pred['action'] == 'select':
                grounded_action = f"select [{pred['idx']}] [{pred['value']}]"
            elif pred['action'] == 'scroll':
                grounded_action = f"scroll [{pred['idx']}]"
            elif pred['action'] == 'goto':
                grounded_action = f"goto [{pred['value']}]"
            elif pred['action'] == 'google_search':
                grounded_action = f"search_google [{pred['value']}]"
            else:
                grounded_action = pred['action']
            
            return grounded_action, pred
            
        except:
            logger.error('Error in parsing the prediction dict {}'.format(action))
            pred = {'idx': 'regex fail', 'action_natural_language': '', 'action': 'regex fail', 'value': 'regex fail'}
            logger.error(traceback.format_exc())
            is_action_valid = False

            return pred['action'], pred
