import argparse
from datasets import Dataset, load_dataset
from evals.in_domain_eval.dataset import WebTrajDataCollator
import os
import json
import logging
import traceback
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from evals.mind2web_orig_eval.utils import create_model, generate_acc_tree, setup_logging
from evals.mind2web_orig_eval.set_of_mark import add_set_of_mark
from evals.mind2web_orig_eval.eval_utils import *
from tqdm import tqdm
import torch
from evals.mind2web_orig_eval.processors import ImageObservationProcessor
from bs4 import BeautifulSoup
import ast
import json
import re
import random
import pickle as pkl

logger = logging.getLogger("__main__")

# Code adapted from: https://github.com/njucckevin/SeeClick

class WebRandomWalkerFusionFlow:
    def __init__(self, args, eval_dataset, annotation_action_idx_dict, scores_all_data):
        self.args = args
        self.scores_all_data = scores_all_data
        self.viewport_size = {'width': args.viewport_width, 'height': args.viewport_height}

        self.eval_dataset = eval_dataset
        self.annotation_action_idx_dict = annotation_action_idx_dict

        self.image_observation_type = 'image_som'
        
        if args.model == 'phi-3.5':
            model_name_or_path = 'microsoft/Phi-3-vision-128k-instruct'
            
            self.processor = AutoProcessor.from_pretrained(
                model_name_or_path, trust_remote_code=True, num_crops=args.num_crops)
            
            self.data_collator = WebTrajDataCollator(self.args, self.processor)

            # for full finetuning, GPU memory can't be cleared (likely caused by deepspeed
            # https://github.com/microsoft/DeepSpeed/issues/3677)
            # so we don't reload the model
            if args.ckpt_path:
                model_name_or_path = args.ckpt_path

            self.model = create_model(
                model_name_or_path,
                use_flash_attention=args.use_flash_attention,
            )
        elif args.model == 'qwen-7b':
            model_name_or_path = "Qwen/Qwen2-VL-7B-Instruct"

            if args.ckpt_path:
                model_name_or_path = args.ckpt_path

            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2",
            )

            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

            self.data_collator = WebTrajDataCollator(self.args, self.processor)

        self.model.eval()

        if not args.no_cuda:
            self.model.to('cuda')

        logging.info(type(self.model))

        self.image_processor = ImageObservationProcessor(
            args, self.image_observation_type, self.viewport_size
        )

    def get_state(self):
        som_image_obs, parsed_html_str = self.image_processor.process_new(self.browser_agent.browser_env.page, self.browser_agent.browser_env.page.client, intent=None)

        html = self.browser_agent.browser_env.page.content()

        return {'page': self.browser_agent.browser_env.page, 'client': self.browser_agent.browser_env.page.client, 'content_str': parsed_html_str, 'image_obs': som_image_obs, 'html': html}

    def eval_episode(self, annotation_id, episode_id, split):

        logging.info('episode_id = {}'.format(episode_id))
        os.makedirs(os.path.join(self.args.log_dir, str(episode_id)), exist_ok=True)

        num_step_in_episode = 0
        history = None
        response_ls = []
        episode_result = {'steps': {}}
        action_history = []

        # for k, step in enumerate(episode['actions']):
        for k, action_id in enumerate(tqdm(self.annotation_action_idx_dict[annotation_id])):
            ex_id = self.annotation_action_idx_dict[annotation_id][action_id]
            example = self.eval_dataset[ex_id]

            # logging.info('example = ', example)
            goal = example['confirmed_task']
            logging.info('goal = {}'.format(goal))

            html_content = example['cleaned_html']
            logging.info('html_content = {}'.format(html_content[:20]))

            # Parse the HTML content
            soup = BeautifulSoup(html_content, 'html.parser')

            # Generate the acc tree
            _, eleid_attr_dict = generate_acc_tree(soup)

            # get the screenshot image
            screenshot = example['screenshot']
            logging.info('size(screenshot) = {}'.format(screenshot.size))
            # logging.info('type(screenshot) = {}'.format(type(screenshot)))

            # SOM annotation
            som_screenshot, visible_rects, _, _, pos_boxes, all_boxes, backend_node_id_to_idx = add_set_of_mark(screenshot, example, self.scores_all_data, use_top50=True, omit_top50_pos=True)
            w, h = som_screenshot.size

            logger.info('backend_node_id_to_idx = {}'.format(backend_node_id_to_idx))

            # cropping
            y_offset = 0

            try:
                pos_boxes = {backend_node_id_to_idx[json.loads(json.loads(candidate)['attributes'])['backend_node_id']]:json.loads(json.loads(candidate)['attributes'])['bounding_box_rect'].split(',') for candidate in example['pos_candidates']}
                keep_idx = list(pos_boxes.keys())[0]
                max_area = -1
                for box_id in pos_boxes:
                    if float(pos_boxes[box_id][2]) * float(pos_boxes[box_id][3]) > max_area:
                        max_area = float(pos_boxes[box_id][2]) * float(pos_boxes[box_id][3])
                        keep_idx = box_id
                
                pos_box = [float(v) for v in pos_boxes[keep_idx]]
                center = [pos_box[0] + pos_box[2] / 2, pos_box[1] + pos_box[3] / 2]
                
                logging.info('pos center = {}'.format(center))

                if center[1] < 720:
                    # crop to 720
                    som_screenshot = som_screenshot.crop((0, 0, w, 720))
                else:
                    logging.info('doing random crop')

                    # crop to random window around the center that covers the whole width
                    x = random.randint(0, 720)
                    som_screenshot = som_screenshot.crop((0, center[1] - x, w, center[1] - x + 720))
                    y_offset = center[1] - x
            except:
                logging.info('No positive boxes found')
                pos_boxes = {}
                # continue

            logging.info('size(som_screenshot) = {}'.format(som_screenshot.size))

            # find the new visible rects after cropping
            visible_rects_crop = set()

            for ele_id in visible_rects:
                for rect in all_boxes[ele_id]["rects"]:
                    # Empty rectangles
                    if not rect:
                        continue
                    if rect["width"] * rect["height"] == 0:
                        continue

                    mid = ((rect["right"] + rect["left"]) / 2.0, (rect["top"] + rect["bottom"]) / 2.0 - y_offset)

                    if 0 <= mid[0] and mid[0] < som_screenshot.size[0]:
                        if mid[1] >= 0 and mid[1] < som_screenshot.size[1]:
                            visible_rects_crop.add(ele_id)
            
            visible_rects_crop = list(visible_rects_crop)
            visible_rects = visible_rects_crop

            logging.info('len(visible_rects) = {}'.format(len(visible_rects)))

            if len(pos_boxes)==0:
                logging.info('No positive boxes found')
                pos_boxes = {}
                # continue
            
            som_screenshot_path = os.path.join(self.args.log_dir, str(episode_id), f'screenshot_som_{k}.png')
            som_screenshot.save(som_screenshot_path)

            backend_node_idx_to_id = {v:k for k, v in backend_node_id_to_idx.items()}

            acc_tree = ''

            acc_tree_elements = visible_rects

            # for ele_id in visible_rects:
            for ele_id in acc_tree_elements:
                tag_name = eleid_attr_dict[backend_node_idx_to_id[ele_id]]['tag_name']
                text_content = eleid_attr_dict[backend_node_idx_to_id[ele_id]]['alt-text']

                acc_tree += f"[{ele_id}] [{tag_name}] [{text_content}]\n"
            
            logging.info('acc_tree = {}'.format(acc_tree))

            episode_result['steps'][f'step_{k}'] = None

            # get prediction
            idx = f'{split}_{episode_id}_{k}'
            
            logging.info('action_history = {}'.format(action_history))
            batch_data = self.data_collator(overall_task=goal, acc_tree=acc_tree, som_screenshot_path=som_screenshot_path, action_history=action_history)

            # generate output
            generation_args = { 
                "max_new_tokens": self.args.max_new_tokens, 
                "temperature": self.args.temperature, 
                "do_sample": True, 
            }
            if self.args.temperature==0:
                generation_args['do_sample'] = False
            
            with torch.no_grad():
                if self.args.model == 'phi-3.5':
                    generate_ids = self.model.generate(**batch_data, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args) 

                    # decode the output

                    # remove input tokens 
                    generate_ids = generate_ids[:, batch_data['input_ids'].shape[1]:]
                    raw_response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
                else:
                    generate_ids = self.model.generate(**batch_data, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args) 

                    # remove input tokens 
                    generate_ids = generate_ids[:, batch_data['input_ids'].shape[1]:]
                    raw_response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

            with open(f'{self.args.log_dir}/action_pred.txt', 'a') as f:
                f.write(idx + str(raw_response) + '\n')

            logging.info('raw_response = {}'.format(raw_response))

            # parse response and map to label_coordinates
            response = None
            try: 
                response = ast.literal_eval(raw_response)
            except:
                logging.info(traceback.format_exc())
                pattern = r'\{"action":.*?\}'
                # Search for the pattern in the input string
                matches = re.findall(pattern, raw_response)
                # Assuming we want the first match
                if matches:
                    response = json.loads(matches[0])
                else:
                    pattern = r'\{"action_type":.*?\}'
                    matches = re.findall(pattern, raw_response)
                    if matches:
                        response = json.loads(matches[0])
            
            try:
                logging.info('response = {}'.format(response))
            except:
                response = {'action': 'fail', 'action_natural_language': 'fail', 'idx': 'fail', 'value':'fail'}
            
            if response:
                action_reprs = example['action_reprs'][k]
                action_op = ast.literal_eval(example["operation"])
                action_type = action_op["op"]

                if action_type == 'CLICK':
                    try:
                        tag, alt_text = action_reprs.split('->')[0].split()
                    except:
                        tag = action_reprs.split('->')[0]
                        alt_text = ''

                    tag = tag.strip().replace('[', '').replace(']', '')
                    alt_text = alt_text.strip()

                    if alt_text == '':
                        new_action_nl = "Click on the {} element".format(tag)
                    else:
                        new_action_nl = "Click on the {} {}".format(alt_text, tag)
                    
                elif action_type == 'SELECT':
                    tag, alt_text = action_reprs.split('->')
                    tag = tag.strip().replace('[select]', '')
                    alt_text = alt_text.strip().replace('SELECT: ', '')
                    new_action_nl = "Select the option {} from the {} list".format(alt_text, tag)
                    
                elif action_type == 'TYPE':
                    tag, alt_text = action_reprs.split('->')
                    tag = tag.strip().replace('[textbox]', '')
                    alt_text = alt_text.strip().replace('TYPE: ', '')
                    new_action_nl = "Type \"{}\" into the {} textbox".format(alt_text, tag)
                
                action_history.append(new_action_nl)

            response_ls.append(response)
                
            # start evaluation
            try:
                action_pred = response
                if action_pred["action"] == 'click':
                    action_pred["action"] = 4
                elif action_pred["action"] == 'select':
                    action_pred["action"] = 2
                elif action_pred["action"] == 'type':
                    action_pred["action"] = 3
                else: # not expected
                    action_pred["action"] = 0

                pos_candidates = example['pos_candidates']

                action_step_ref = self.action2step(annotation_id, action_id)
                action_step_pos_ele_ids = list(pos_boxes.keys())
                
                logging.info('action_step_pos_ele_ids = {}'.format(action_step_pos_ele_ids))
                logging.info('action_step_ref = {}'.format(action_step_ref))

                action_step_ref = ast.literal_eval(action_step_ref)
                step_result = {"idx": idx, "sentence": action_pred, 'ground_truth': action_step_ref,
                            "Op_match": False, "Ele_match": False, "Op_F1": [0, action_step_ref["action_type"]]}

                if action_pred["action"] == action_step_ref["action_type"]:
                    step_result["Op_match"] = True


                if action_pred["idx"] in pos_boxes:
                    step_result["Ele_match"] = True
                
                logging.info('type(step_result) = {}'.format(type(step_result)))
                logging.info('step_result = {}'.format(step_result))

                # In mind2web, action is converted into a string, that is, if it is TYPE, F1 between characters needs to be considered.
                pred_str = str(action_pred["action"])
                if action_pred["action"] == 3 or action_pred["action"] == 2:
                    pred_str += ' '
                    pred_str += action_pred["value"].lower()
                ref_str = str(action_step_ref["action_type"])
                if action_step_ref["action_type"] == 3 or action_step_ref["action_type"] == 2:
                    ref_str += ' '
                    ref_str += action_step_ref["value"].lower()

                op_f1 = calculate_f1(pred_str, ref_str)
                step_result["Op_F1"][0] = op_f1

                logging.info('step_result = {}'.format(step_result))
                num_step_in_episode += 1

                episode_result['steps'][f'step_{k}'] = step_result
                with open(f'{self.args.log_dir}/step_results.json', 'a') as f:
                    json.dump(step_result, f)
                    f.write('\n')
                
            except:
                logging.info(traceback.format_exc())
                logging.info("format wrong!!!\n\n\n")

                action_step_ref = self.action2step(annotation_id, action_id)
                action_step_ref = ast.literal_eval(action_step_ref)
                step_result = {"idx": idx, "sentence": action_pred, 'ground_truth': action_step_ref,
                            "Op_match": False, "Ele_match": False, "Op_F1": [0, action_step_ref["action_type"]]}
                
                episode_result['steps'][f'step_{k}'] = step_result
                with open(f'{self.args.log_dir}/step_results.json', 'a') as f:
                    json.dump(step_result, f)
                    f.write('\n')

                num_step_in_episode += 1

            # logging.info(step_result)

        # assert num_step_in_episode == len(episode_result['steps'])
        # num_step_in_episode = len(episode_result['steps'])
        ele_match = [step['Ele_match'] for step in episode_result['steps'].values() if step]
        op_f1 = [step['Op_F1'][0] for step in episode_result['steps'].values() if step]
        step_success = [step['Op_F1'][0]==1 and step['Ele_match'] for step in episode_result['steps'].values() if step]
        num_step_in_episode = len(ele_match)
        if num_step_in_episode == 0:
            return episode_result
        episode_result['avg_ele_match'] = sum(ele_match) / num_step_in_episode
        episode_result['avg_op_f1'] = sum(op_f1) / num_step_in_episode
        episode_result['step_SR'] = sum(step_success) / num_step_in_episode

        logging.info('episode_result = ', episode_result)

        with open(f'{self.args.log_dir}/episode_results.json', 'a') as f:
            json.dump(episode_result, f)
            f.write('\n')
        logging.info("step_SR: " + str(episode_result['step_SR']))
        return episode_result

    # convert action to prediction format (and return the groundtruth bbox)
    def action2step(self, annotation_id, action_id):

        ex_id = self.annotation_action_idx_dict[annotation_id][action_id]
        example = self.eval_dataset[ex_id]
        
        # logging.info('example = {}'.format(example))
        logging.info('example.keys() = {}'.format(example.keys()))
        action_op = ast.literal_eval(example["operation"])
        logging.info('action_op.keys() = {}'.format(action_op.keys()))

        action_type = action_op["original_op"]
        assert action_type in ['CLICK', 'TYPE', 'SELECT', 'HOVER', 'ENTER']

        if action_type in ['CLICK', 'HOVER', 'ENTER']:
            action_step = "{{\"action_type\": {}}}".format(4)
        elif action_type == 'SELECT':
            select_value = action_op["value"]

            # Escaping double quotes
            select_value = select_value.replace('"', '\\"')

            action_step = "{{\"action_type\": {}, \"value\": \"{}\"}}".format(2, select_value)
        elif action_type == 'TYPE':
            typed_text = action_op["value"]

            # Escaping double quotes
            typed_text = typed_text.replace('"', '\\"')

            action_step = "{{\"action_type\": {}, \"value\": \"{}\"}}".format(3, typed_text)
            
        # action_step = "{{\"action_type\": {}, \"click_point\": {}}}".format(action_type, click_point)

        return action_step
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['phi-3.5', 'qwen-7b'], default='phi-3.5')
    parser.add_argument('--max-steps', type=int, default=5, help='Maximum number of steps to simulate')
    parser.add_argument('--print-parsed-tree', action='store_true', help='Print the parsed tree in stdout')
    parser.add_argument('--seed', type=int, default=736537, help='Random seed')
    parser.add_argument('--use-flash-attention', action='store_true', help='use flash attention', default=False)
    parser.add_argument('--ckpt-path', type=str, default=None, help='Path to the model checkpoint')
    parser.add_argument('--bf16', action='store_true', help='Use bf16 precision', default=False)
    parser.add_argument('--log-dir', type=str, default='toy/', help='Path to the logging dir')
    parser.add_argument('--task-field', type=str, choices=['summary_abstract', 'task_summary'], default='task_summary', help='Field to use for the task description')
    parser.add_argument('--split', type=str, choices=['test_domain', 'test_task', 'test_website'], default='test_task', help='test split')
    parser.add_argument('--viewport-width', type=int, default=1280, help='viewport width')
    parser.add_argument('--viewport-height', type=int, default=720, help='viewport height')

    parser.add_argument('--num_crops', type=int, default=16, help='Number of maximum image crops')

    parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.01, help='Generation temperature')

    parser.add_argument('--n-ex-subset', type=int, default=100, help='Number of examples to use in test split')
    parser.add_argument('--no-cuda', action='store_true', help='Use CPU')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--score-file', type=str, default='/home/pahuja.9/research_nfs/web_traj_gen/scores_all_data.pkl', help='Path to the deberta scores for top 50 candidates')

    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    setup_logging(args.log_dir)

    logging.info(args)

    # Create the dataset
    ds = load_dataset("osunlp/Multimodal-Mind2Web")

    eval_dataset = ds[args.split]

    total = 0

    annotation_action_idx_dict = {}

    for ex_id, example in enumerate(tqdm(eval_dataset)):
        annotation_id = example['annotation_id']
        action_uid = example['action_uid']

        if annotation_id not in annotation_action_idx_dict:
            annotation_action_idx_dict[annotation_id] = {}
        
        annotation_action_idx_dict[annotation_id][action_uid] = ex_id

    # logging.info('annotation_action_idx_dict = ', annotation_action_idx_dict)

    scores_all_data = pkl.load(open(args.score_file, 'rb'))

    flow = WebRandomWalkerFusionFlow(args, eval_dataset, annotation_action_idx_dict, scores_all_data)

    total_sr = 0
    total_ele_match = 0
    total_op_f1 = 0

    count = 0

    for episode_id, annotation_id in enumerate(tqdm(annotation_action_idx_dict)):

        try:
            episode_result = flow.eval_episode(annotation_id, episode_id, args.split)

            total_sr += episode_result['step_SR']
            total_ele_match += episode_result['avg_ele_match']
            total_op_f1 += episode_result['avg_op_f1']
            count += 1
        except:
            logging.info(traceback.format_exc())
            logging.info("Error in episode {}".format(episode_id))
            # break
        
        if args.debug:
            break

    average_sr = total_sr*100.0 / count
    average_ele_match = total_ele_match*100.0 / count
    average_op_f1 = total_op_f1*100.0 / count

    print(f"average step_SR = {average_sr:.2f}")
    print(f"average ele_match = {average_ele_match:.2f}")
    print(f"average op_f1 = {average_op_f1:.2f}")

            
