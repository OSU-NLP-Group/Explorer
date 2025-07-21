import argparse
import random

import computergym
import gym
from .slm_agent_qwen import SLMAgent

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from evals.mind2web_orig_eval.utils import create_model, generate_acc_tree, setup_logging
from .processors import ImageObservationProcessor
from evals.miniwob.browser_env import ScriptBrowserEnv
# from in_domain_eval.dataset import WebTrajDataCollator
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM
from .actions import create_id_based_action, create_none_action

import sys
import os, json
import torch
from PIL import Image
import logging
import ast
import numpy as np
import traceback
import re

np.bool8 = np.bool_

logging.basicConfig(level=logging.INFO)


SYSTEM_MESSAGE = '''You are an expert at completing instructions on Webpage screens. 
               You will be presented with a screenshot image with some numeric tags.
               If you decide to click somewhere, you should choose the numeric element idx that is the closest to the location you want to click.  
               You should decide the action to continue this instruction.
               You will be given the accessibility tree of the current screen in the format: '[element_idx] [role] [alt text or button name]'.
               Here are the available actions:
{"action": "click", "action_natural_language": str, "idx": <element_idx>}
{"action": "type", "action_natural_language": str, "idx": <element_idx>, "value": <the text to enter>}
{"action": "select", "action_natural_language": str, "idx": <element_idx>, "value": <the option to select>}
Your final answer must be in the above format.
'''
SYSTEM_MESSAGE_NEW = '''You are an expert at completing instructions on Webpage screens. 
               You will be presented with a screenshot image with some numeric tags.
               If you decide to click somewhere, you should choose the numeric element idx that is the closest to the location you want to click.  
               You should decide the action to continue this instruction.
               You will be given the accessibility tree of the current screen in the format: '[element_idx] [role] [alt text or button name]'.
               Here are the available actions:
{"action": "click", "action_natural_language": str, "idx": <element_idx>}
{"action": "type", "action_natural_language": str, "idx": <element_idx>, "value": <the text to enter>}
{"action": "select", "action_natural_language": str, "idx": <element_idx>, "value": <the option to select>}
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
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, overall_task, acc_tree, som_screenshot, action_history=[], last_action = ''):
        
        # system message
        system_message = {
            'role': 'system',
            # 'content': SYSTEM_MESSAGE,
            'content': SYSTEM_MESSAGE_NEW,
        }
        
        # if len(action_history)>0:
        #     last_action = action_history[-1]
        # else:
        #     last_action = ''

        user_prompt = USER_MESSAGE.format(overall_task, action_history, acc_tree)
        image = Image.fromarray(som_screenshot)

        messages_new = []
        messages_new.append(system_message)

        user_prompt_0, user_prompt_1 = user_prompt.split('<|image_1|>')
        messages_new.append({"role": "user", "content": [{"type": "text", "text": user_prompt_0}, {"type": "image"}, {"type": "text", "text": user_prompt_1}]})
    
        prompt = self.processor.apply_chat_template(messages_new, add_generation_prompt=True)
        batch = self.processor(text=prompt, images=[image], padding=True, return_tensors="pt")
        batch.to('cuda')
        
        return batch

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="click-button")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--llm", type=str, default="chatgpt")
    parser.add_argument("--erci", type=int, default=0)
    parser.add_argument("--step", type=int, default=-1)

    parser.add_argument("--irci", type=int, default=1)
    parser.add_argument("--seed", type=int, default=73631)
    parser.add_argument("--sgrounding", action="store_true", default=False)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--use-dynamic-seed", action="store_true", default=False)

    parser.add_argument('--use-async-playwright', action='store_true', help='use async playwright', default=False)
    parser.add_argument('--record-video', action='store_true', help='record video in log dir')
    parser.add_argument('--record-trace', action='store_true', help='record trace in log dir')
    parser.add_argument('--ckpt-path', type=str, default='/home/pahuja.9/research_nfs/web_traj_gen/ckpts/phi3.5_s10_m2wtrain_sample_10epoch/', help='Path to the model checkpoint')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.01, help='Generation temperature')
    parser.add_argument("--parse-two-dicts", action="store_true", help='parse the first dict if 2 are predicted')
    parser.add_argument('--output-dir', type=str, help='output dir. for screenshots', default='toy/')
    parser.add_argument("--no-multiple-parse", action="store_true", help='do NOt use execute_action()')
    parser.add_argument("--add-class", action="store_true", help='add class to acc tree')
    parser.add_argument("--omit-empty-div", action="store_true", help='omit empty div elements in acc tree and SOM')
    parser.add_argument("--use-dict-last-action", action="store_true", help='use dict for last action repetition avoidance')
    parser.add_argument("--add-class-subset", action="store_true", help='add class for subset of tasks')

    opt = parser.parse_args()

    if opt.add_class_subset and opt.env in ['click-checkboxes-transfer', 'click-dialog-2', 'count-shape', 'email-inbox', 'enter-text', 'enter-time']:
        print('add class activated')
        opt.add_class = True

    return opt


def get_html_state_from_real(driver, opt):
    if opt.env == "facebook":
        main_html_xpath = '//*[@id="content"]'
        html_body = driver.find_element(By.XPATH, main_html_xpath).get_attribute(
            "outerHTML"
        )
    else:
        raise NotImplemented

    return html_body


def perform_instruction(driver, instruction):
    instruction = instruction.split(" ")
    inst_type = instruction[0]
    inst_type = inst_type.lower()

    if inst_type == "type":
        characters = " ".join(instruction[1:])
        characters = characters.replace('"', "")
        chain = ActionChains(driver)
        chain.send_keys(characters)
        chain.perform()
    elif inst_type == "clickxpath":
        xpath = " ".join(instruction[1:])
        element = driver.find_element(By.XPATH, str(xpath))
        chain = ActionChains(driver)
        chain.move_to_element(element).click().perform()
    elif inst_type == "press":
        key_type = instruction[1]
        if key_type == "enter":
            chain = ActionChains(driver)
            chain.send_keys("\n")
            chain.perform()
        elif key_type == "space":
            chain = ActionChains(driver)
            chain.send_keys(" ")
            chain.perform()
        else:
            raise NotImplemented
    else:
        raise ValueError("Invalid instruction")


def get_webdriver(url):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("disable-gpu")
    options.add_argument("no-sandbox")

    driver = webdriver.Chrome(chrome_options=options)
    driver.implicitly_wait(5)
    driver.maximize_window()
    driver.implicitly_wait(5)

    driver.get(url)
    driver.implicitly_wait(10)
    return driver

def execute_action(response, rects, action_history, action_history_complete, slm_agent, env, states, browser_env, image_processor, add_class):
    action_history.append(response['action_natural_language'])
    action_history_complete.append(str(response))

    try:
        goal = states[0].utterance
    except:
        return None, None, browser_env, env, states, rects, [0], [False]
    
    if response['action'] in ['click', 'select']:
        
        try:
            print('rects = {}'.format(rects))

            # get xpath from raw_response
            tag_idx = str(response['idx'])

            if tag_idx not in rects:
                return None, None, browser_env, env, states, rects, [0], [False]
            
            xpath = rects[tag_idx]['xpath']
            
            instruction = f'clickxpath {xpath}'

            print(f'instruction = {instruction}')
            
            print('flag 4')
            # instruction = llm_agent.generate_action()
            logging.info(f"The executed instruction: {instruction}")

            miniwob_action = slm_agent.convert_to_miniwob_action(instruction)

            states, rewards, dones, _ = env.step([miniwob_action])
        except:
            print("Invalid action or rci action fail")
            rewards = [0]
            dones = [False]
            # break
            traceback.format_exc()
            # sys.exit(0)

    elif response['action'] == 'type':
        tag_idx = str(response['idx'])

        if tag_idx not in rects:
            return None, None, browser_env, env, states, rects, [0], [False]
        
        xpath = rects[tag_idx]['xpath']

        instruction = f'clickxpath {xpath}'
        # execute clickxpath followed by type instruction
        print(f'instruction = {instruction}')

        try:
            # instruction = llm_agent.generate_action()
            logging.info(f"The executed instruction: {instruction}")

            miniwob_action = slm_agent.convert_to_miniwob_action(instruction)

            states, rewards, dones, _ = env.step([miniwob_action])
        except:
            # print("Invalid action or rci action fail")
            rewards = [0]
            dones = [False]
            # break
            traceback.format_exc()
        
        try:
            text_to_type = response['value']
            instruction = f'type {text_to_type}'
            print(f'instruction = {instruction}')

            # instruction = llm_agent.generate_action()
            logging.info(f"The executed instruction: {instruction}")

            miniwob_action = slm_agent.convert_to_miniwob_action(instruction)

            states, rewards, dones, _ = env.step([miniwob_action])
        except:
            # print("Invalid action or rci action fail")
            rewards = [0]
            dones = [False]
            # break
            traceback.format_exc()

    else:
        # return NotImplementedError
        instruction = ''
        rewards = [0]
        dones = [False]

    print(f'rewards = {rewards}, dones = {dones}')

    if rewards[0] > 0:
        return None, None, browser_env, env, states, rects, rewards, dones

    try:
        if response['action'] in ['type']:
            assert 'value' in response
        if response['action'] in ['click', 'type']:
            assert 'idx' in response
        
        if response['action'] == 'type':
            grounded_action = f"type [{response['idx']}] [{response['value']}]"
        elif response['action'] == 'click':
            grounded_action = f"click [{response['idx']}]"
        elif response['action'] == 'select':
            if 'value' in response:
                grounded_action = f"select [{response['idx']}] [{response['value']}]"
            else:
                grounded_action = f"click [{response['idx']}]"
        else:
            grounded_action = response['action']

        cur_action = create_id_based_action(grounded_action)
    except:
        logging.error('Action parsing error')
        # traceback.print_exc()
        logging.error(traceback.format_exc())

        cur_action = create_none_action()

    logging.info(f"Action to be executed: {cur_action}")

    try:
        browser_env.step(cur_action)
    except:
        traceback.print_exc()
        
    # transform html to acc tree
    som_image_obs, acc_tree, rects = image_processor.process_new(browser_env.page, browser_env.page.client, intent=None, add_class=add_class, omit_task_desc_ele=True, goal=goal, opt=opt)

    return som_image_obs, acc_tree, browser_env, env, states, rects, rewards, dones

def miniwob(opt):
    env = gym.make("MiniWoBEnv-v0", env_name=opt.env, headless=opt.headless)

    max_step = 10

    image_observation_type = 'image_som'
    viewport_size = {'width': 1280, 'height': 1080}

    image_processor = ImageObservationProcessor(
            opt, image_observation_type, viewport_size
        )

    # create a new browser env
    browser_env = ScriptBrowserEnv(opt, browser_type='chrome', viewport_size=viewport_size, image_processor=image_processor)
    browser_env.setup('https://www.google.com', None)

    model_name_or_path = "Qwen/Qwen2-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    data_collator = WebTrajDataCollator(processor)
    
    success = 0
    for episode_id in range(opt.num_episodes):
        slm_agent = SLMAgent(
            opt,
            opt.env,
            ckpt_path=opt.ckpt_path,
            rci_plan_loop=opt.erci,
            rci_limit=opt.irci,
            llm=opt.llm,
            state_grounding=opt.sgrounding,
        )
        # initialize environment
        if opt.use_dynamic_seed:
            seed = random.random()
            print(f'seed = {seed}')
            states = env.reset(seeds=[seed], record_screenshots=True)
        else:
            states = env.reset(seeds=[opt.seed], record_screenshots=True)

        html_state = get_html_state(opt, states)
        goal = states[0].utterance

        print(html_state)

        try:
            browser_env.page.set_content(html_state)
        except:
            print('goto timeout')
        
        # transform html to acc tree
        som_image_obs, acc_tree, rects = image_processor.process_new(browser_env.page, browser_env.page.client, intent=None, add_class=opt.add_class, omit_task_desc_ele=True, goal=goal, opt=opt)

        Image.fromarray(som_image_obs).save(os.path.join(opt.output_dir, f'{episode_id}_0.png'))
        

        step = max_step
        rewards = []
        action_history = []
        action_history_complete = []

        logging.info(f"The number of generated action steps: {step}")
        goal = states[0].utterance
        print(f'goal = {goal}')

        step_id = 0

        while step_id < step:
            assert len(states) == 1
            
            print(f'step_id = {step_id}, action_history = {action_history}')
            print(f'acc_tree = {acc_tree}')

            if opt.use_dict_last_action:
                if len(action_history_complete) > 0:
                    last_action = action_history_complete[-1]
                else:
                    last_action = ''
            else:
                if len(action_history) > 0:
                    last_action = action_history[-1]
                else:
                    last_action = ''

            batch_data = data_collator(overall_task=goal, acc_tree=acc_tree, som_screenshot=som_image_obs, action_history=action_history, last_action=last_action)

            # generate output
            generation_args = { 
                "max_new_tokens": opt.max_new_tokens, 
                "temperature": opt.temperature, 
                "do_sample": True, 
            }
            
            with torch.no_grad():
                generate_ids = slm_agent.model.generate(**batch_data, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 

                # decode the output
                # remove input tokens 
                generate_ids = generate_ids[:, batch_data['input_ids'].shape[1]:]
                raw_response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
                
                print(f'raw_response = {raw_response}')
            
            if not opt.no_multiple_parse:
                response_dict_strs = re.findall(r'\{.*?\}', raw_response)
                
                response_dict_strs_new = []

                for x in response_dict_strs:
                    fixed_str = re.sub(r"(?<=\{|,)\s*'([^']+)'(?=\s*:)", r'"\1"', x)  # Keys
                    fixed_str = re.sub(r":\s*'([^']*)'", r': "\1"', fixed_str)  # String values
                    response_dict_strs_new.append(fixed_str)

                print(f'response_dict_strs = {response_dict_strs_new}')

                try:
                    response_dicts = [json.loads(d) for d in response_dict_strs_new]
                except:
                    response_dicts = [json.loads(d.replace('"', '').replace("'", '"')) for d in response_dict_strs]


                if len(response_dicts)==0:
                    step_id += 1
                
                dones = [False]
                
                if len(response_dicts)==1 and isinstance(response_dicts[0]['idx'], list) and len(response_dicts[0]['idx'])>1: # handle click-shades
                    response_dict = response_dicts[0]
                    response_dicts = [response_dict.copy() for _ in range(len(response_dicts[0]['idx']))]

                    for i, response_dict in enumerate(response_dicts):
                        response_dict['idx'] = response_dict['idx'][i]

                    print('response_dicts = {}'.format(response_dicts))

                for response_dict in response_dicts:
                    logging.info('response_dict = {}'.format(response_dict))
                    som_image_obs_new, acc_tree_new, browser_env, env, states, rects, rewards, dones = execute_action(response_dict, rects, action_history, action_history_complete, slm_agent, env, states, browser_env, image_processor, add_class=opt.add_class)

                    if som_image_obs_new is not None:
                        som_image_obs = som_image_obs_new

                    if acc_tree_new is not None:
                        acc_tree = acc_tree_new
                    
                    if rewards[0] > 0:
                        break

                    if all(dones) and rewards[0] > 0:  # or llm_agent.check_finish_plan():
                        break

                    step_id += 1

                    img = Image.fromarray(som_image_obs)
                    img.save(os.path.join(opt.output_dir, f'{episode_id}_{step_id}.png'))

                if all(dones) and rewards[0] > 0:  # or llm_agent.check_finish_plan():
                    break

            else:
                # parse response and map to label_coordinates
                response = None
                try: 
                    response = ast.literal_eval(raw_response)
                except:
                    print('raw_response = {}'.format(raw_response))
                    logging.info(traceback.format_exc())

                    if opt.parse_two_dicts:
                        # Regex to extract the first dictionary
                        match = re.search(r'(\{.*?\})', raw_response, re.DOTALL)

                        if match:
                            first_dict_str = match.group(1)
                            response = ast.literal_eval(first_dict_str)
            
                # get action type and id from raw_response
                if response:
                    action_history.append(response['action_natural_language'])
                    action_history_complete.append(str(response))

                    if response['action'] in ['click', 'select']:
                        
                        try:
                            print('rects = {}'.format(rects))
                        
                            # get xpath from raw_response
                            tag_idx = str(response['idx'])
                            xpath = rects[tag_idx]['xpath']
                            
                            instruction = f'clickxpath {xpath}'

                            print(f'instruction = {instruction}')
                            
                            # instruction = llm_agent.generate_action()
                            logging.info(f"The executed instruction: {instruction}")

                            miniwob_action = slm_agent.convert_to_miniwob_action(instruction)

                            states, rewards, dones, _ = env.step([miniwob_action])
                        except:
                            # print("Invalid action or rci action fail")
                            rewards = [0]
                            dones = [False]
                            # break
                            traceback.format_exc()

                    elif response['action'] == 'type':
                        tag_idx = str(response['idx'])
                        xpath = rects[tag_idx]['xpath']

                        instruction = f'clickxpath {xpath}'
                        # execute clickxpath followed by type instruction
                        print(f'instruction = {instruction}')

                        try:
                            # instruction = llm_agent.generate_action()
                            logging.info(f"The executed instruction: {instruction}")

                            miniwob_action = slm_agent.convert_to_miniwob_action(instruction)

                            states, rewards, dones, _ = env.step([miniwob_action])
                        except:
                            rewards = [0]
                            dones = [False]
                            traceback.format_exc()
                        
                        try:
                            text_to_type = response['value']
                            instruction = f'type {text_to_type}'
                            print(f'instruction = {instruction}')

                            logging.info(f"The executed instruction: {instruction}")

                            miniwob_action = slm_agent.convert_to_miniwob_action(instruction)

                            states, rewards, dones, _ = env.step([miniwob_action])
                        except:
                            # print("Invalid action or rci action fail")
                            rewards = [0]
                            dones = [False]
                            # break
                            traceback.format_exc()

                    else:
                        instruction = ''
                        pass

                    print(f'rewards = {rewards}')

                    if rewards[0] > 0:
                        break

                    if all(dones) and rewards[0] > 0:  # or llm_agent.check_finish_plan():
                        break

                    html_state = get_html_state(opt, states)
                    
                    try:
                        if response['action'] in ['type', 'select']:
                            assert 'value' in response
                        if response['action'] in ['click', 'type']:
                            assert 'idx' in response
                        
                        if response['action'] == 'type':
                            grounded_action = f"type [{response['idx']}] [{response['value']}]"
                        elif response['action'] == 'click':
                            grounded_action = f"click [{response['idx']}]"
                        elif response['action'] == 'select':
                            if 'value' in response:
                                grounded_action = f"select [{response['idx']}] [{response['value']}]"
                            else:
                                grounded_action = f"click [{response['idx']}]"
                        else:
                            grounded_action = response['action']

                        cur_action = create_id_based_action(grounded_action)
                    except:
                        logging.error('Action parsing error')
                        # traceback.print_exc()
                        logging.error(traceback.format_exc())

                        cur_action = create_none_action()

                    logging.info(f"Action to be executed: {cur_action}")

                    browser_env.step(cur_action)
                    
                    # transform html to acc tree
                    som_image_obs, acc_tree, rects = image_processor.process_new(browser_env.page, browser_env.page.client, intent=None, add_class=opt.add_class, omit_task_desc_ele=True, goal=goal, opt=opt)
                
                    step_id += 1
                else:
                    step_id += 1
                    
        if len(rewards) > 0 and rewards[0] > 0:
            success += 1
            # slm_agent.save_result(True)
        else:
            pass
            # slm_agent.save_result(False)
        
        # print(f"success rate: {success / opt.num_episodes}")
        print(f"success rate: {success} / {episode_id + 1} = {success / (episode_id + 1)}")
        print('\n\n')

    print(f"overall success rate: {success / opt.num_episodes}")

    env.close()


def get_html_state(opt, states):
    extra_html_task = [
        "click-dialog",
        "click-dialog-2",
        "use-autocomplete",
        "choose-date",
    ]

    html_body = states[0].html_body
    if opt.env in extra_html_task:
        html_body += states[0].html_extra
    return html_body


if __name__ == "__main__":
    opt = parse_opt()
    if opt.env == "facebook":
        url = "https://www.facebook.com/"
        web(opt, url)
    else:
        miniwob(opt)
        