import argparse
from datasets import Dataset
from in_domain_eval.dataset import WebTrajDataCollator
from in_domain_eval.browser_agent import ScriptBrowserEnvAgent
import os
import json
import logging
import traceback
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, Qwen2VLForConditionalGeneration
from in_domain_eval.utils import create_model
from tqdm import tqdm
import torch
from in_domain_eval.trajectory_verifier import TrajectoryVerifierAgent
from in_domain_eval.gpt_task_completion_agent import GptTaskCompletionAgent
import requests

from in_domain_eval.processors import ImageObservationProcessor
from in_domain_eval.utils import setup_logging

def create_huggingface_dataset(eval_data):
    # Convert the list of dictionaries to a Hugging Face Dataset
    dataset = Dataset.from_list(eval_data)
    return dataset

class WebRandomWalkerFusionFlow:
    def __init__(self, args, async_openai_client=None):
        self.args = args

        self.viewport_size = {'width': args.viewport_width, 'height': args.viewport_height}
        self.image_observation_type = 'image_som'
        
        self.browser_agent = ScriptBrowserEnvAgent(args, browser_type='chrome', viewport_size=self.viewport_size)

        if not args.use_gpt_agent:
            # model_name_or_path = 'microsoft/Phi-3-vision-128k-instruct'
            if args.model == 'phi-3.5':
                model_name_or_path = 'microsoft/Phi-3.5-vision-instruct'

                self.processor = AutoProcessor.from_pretrained(
                    model_name_or_path, trust_remote_code=True, num_crops=args.num_crops)
            
                self.data_collator = WebTrajDataCollator(self.args, self.processor)

                if args.use_lora:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name_or_path,
                        # Phi-3-V is originally trained in bf16 + flash attn
                        # For fp16 mixed precision training, load in f32 to avoid hf accelerate error
                        torch_dtype=torch.bfloat16 if args.use_flash_attention else torch.float32,
                        trust_remote_code=True,
                        _attn_implementation='flash_attention_2' if args.use_flash_attention else 'eager',
                    )
                    if args.ckpt_path:
                        self.model.load_adapter(args.ckpt_path)

                        logging.info('Loaded model from {}'.format(args.ckpt_path))
                        print('Loaded model from {}'.format(args.ckpt_path))
                else:
                    # for full finetuning, GPU memory can't be cleared (likely caused by deepspeed
                    # https://github.com/microsoft/DeepSpeed/issues/3677)
                    # so we don't reload the model
                    if args.ckpt_path:
                        model_name_or_path = args.ckpt_path

                    self.model = create_model(
                        model_name_or_path,
                        use_flash_attention=args.use_flash_attention,
                        use_qlora=args.use_qlora,
                    )
                    
            else:
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
            self.model.to('cuda')

        self.init_setup_error = False
        
        self.image_processor = ImageObservationProcessor(
            args, self.image_observation_type, self.viewport_size
        )

        self.verifier_agent = TrajectoryVerifierAgent(args)
        self.gpt_task_completion_agent = GptTaskCompletionAgent(args)

    def get_state(self):
        som_image_obs, parsed_html_str = self.image_processor.process_new(self.browser_agent.browser_env.page, self.browser_agent.browser_env.page.client, intent=None)

        # som_image_obs, parsed_html_str = self.image_processor.process(self.browser_env.page)
        html = self.browser_agent.browser_env.page.content()

        return {'page': self.browser_agent.browser_env.page, 'client': self.browser_agent.browser_env.page.client, 'content_str': parsed_html_str, 'image_obs': som_image_obs, 'html': html}

    def run_task(self, example, ex_log_dir):
        overall_task = example[args.task_field]
        print(overall_task)

        task_trajectory_data = {'actions': []}
        action_history = []

        # navigate to init_url
        self.browser_agent.browser_env.setup(example['init_url'])

        for step in range(args.max_steps):
            logging.info(f"Step {step}:\n")
            try:
                browser_env_state = self.get_state()
            except:
                logging.info('Error in getting state, exiting...')
                logging.info(traceback.format_exc())
                break

            img = Image.fromarray(browser_env_state['image_obs'])
            img.save(os.path.join(ex_log_dir, f'screenshot_som_{step}.png'))
            acc_tree = browser_env_state['content_str']
            action = {}

            if args.print_parsed_tree:
                logging.info("acc_tree: {}\n".format(acc_tree))

            som_screenshot_path = os.path.join(ex_log_dir, f'screenshot_{step}.png')
            try:
                self.browser_agent.browser_env.page.screenshot(path=som_screenshot_path)
            except:
                continue
            logging.info('history = {}'.format(action_history))
            
            with torch.no_grad():
                if args.use_gpt_agent:
                    new_action = self.gpt_task_completion_agent.act(overall_task, acc_tree, som_screenshot_path, action_history)
                else:
                    batch_data = self.data_collator(overall_task=overall_task, acc_tree=acc_tree, som_screenshot_path=som_screenshot_path, action_history=action_history)

                    # generate output
                    generation_args = { 
                        "max_new_tokens": self.args.max_new_tokens, 
                        "temperature": self.args.temperature, 
                        "do_sample": True, 
                    }
                    if self.args.model == 'phi-3.5':
                        generate_ids = self.model.generate(**batch_data, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args) 

                        # decode the output

                        # remove input tokens 
                        generate_ids = generate_ids[:, batch_data['input_ids'].shape[1]:]
                        new_action = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
                    else:
                        generate_ids = self.model.generate(**batch_data, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args) 

                        # remove input tokens 
                        generate_ids = generate_ids[:, batch_data['input_ids'].shape[1]:]
                        new_action = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

                logging.info("model pred: {}\n".format(new_action))

                # take action using the browser
                try:
                    pred, res, is_action_valid = self.browser_agent.act(new_action, browser_env_state)
                except:
                    continue

                if 'action_natural_language' in pred:
                    action_history.append(pred['action_natural_language'])
                    logging.info('flag 1')
                else:
                    action_history.append('')

                try:
                    action['step_action_nl'] = pred['action_natural_language']
                    logging.info("predicted action: {}\n".format(action['step_action_nl']))
                except:
                    logging.info('pred = {}'.format(pred))
                    action['step_action_nl'] = ''


                logging.info("is_action_valid: {}\n".format(is_action_valid))
                # logging.info("res: {}\n".format(res))
                logging.info('URL: {}'.format(self.browser_agent.browser_env.page.url))

                task_trajectory_data['actions'].append(action)

                
        
        # run verifier
        history = [action['step_action_nl'] for action in task_trajectory_data['actions']]
        img_path = os.path.join(ex_log_dir, 'screenshot_final.png')

        logging.info('history = {}'.format(history))

        self.browser_agent.browser_env.page.screenshot(path=img_path)

        # self.browser_agent.browser_env.close()

        # verifier_agent_response = self.verifier_agent.act(overall_task, history, img_path)
        if self.args.use_single_screenshot_verifier:
            screenshot_history = img_path
        else:
            screenshot_history = [os.path.join(ex_log_dir, f'screenshot_{i}.png') for i in range(step+1)] + [img_path]

        last_page_url = 'https://r.jina.ai/' + self.browser_agent.browser_env.page.url

        # Send a GET request to the URL
        response = requests.get(last_page_url)

        # Ensure the request was successful
        try:
            response.raise_for_status()

            last_page_md = response.content.decode('utf-8')
        except:
            last_page_md = None

        verifier_agent_response = self.verifier_agent.act(overall_task, history, screenshot_history, last_page_md)
        
        logging.info('verifier_agent_response = {}'.format(verifier_agent_response))
        
        is_success = verifier_agent_response.split('\nStatus: ')[-1].lower() == 'success'

        return is_success

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True, help='Input file containing evaluation JSON files')
    parser.add_argument('--max-steps', type=int, default=7, help='Maximum number of steps to simulate')
    parser.add_argument('--print-parsed-tree', action='store_true', help='Print the parsed tree in stdout')
    parser.add_argument('--seed', type=int, default=736537, help='Random seed')
    parser.add_argument('--viewport-width', type=int, default=1280, help='viewport width')
    parser.add_argument('--viewport-height', type=int, default=720, help='viewport height')
    parser.add_argument('--api-auth-type', type=str, choices=['azurecli', 'managed'], default='azurecli', help='API authentication type')
    parser.add_argument('--endpoint', type=str, choices=["https://yadaoai.openai.azure.com/", "https://dataoai2.openai.azure.com/", "https://dataoai3.openai.azure.com/"], default="https://yadaoai.openai.azure.com/", help='API endpoint')
    parser.add_argument('--deployment', type=str, choices=['gpt-4o', 'dataoai2-gpt4', 'gpt4o_2'], default='gpt-4o', help='API model deployment')
    parser.add_argument('--api-version', type=str, default='2024-02-01', help='GPT-4 API version')
    parser.add_argument('--use-async-playwright', action='store_true', help='use async playwright', default=False)
    parser.add_argument('--use-lora', action='store_true', help='use LoRA model', default=False)
    parser.add_argument('--use-flash-attention', action='store_true', help='use flash attention', default=False)
    parser.add_argument('--use-qlora', action='store_true', help='use QLoRA model', default=False)
    parser.add_argument('--ckpt-path', type=str, default=None, help='Path to the model checkpoint')
    parser.add_argument('--bf16', action='store_true', help='Use bf16 precision', default=False)
    parser.add_argument('--log-dir', type=str, default='toy/', help='Path to the logging dir')
    parser.add_argument('--task-field', type=str, choices=['summary_abstract', 'task_summary'], default='task_summary', help='Field to use for the task description')

    parser.add_argument('--num_crops', type=int, default=16, help='Number of maximum image crops')

    parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.01, help='Generation temperature')

    parser.add_argument('--use-gpt-correction', action='store_true', help='Use GPT correction for parsing', default=False)
    parser.add_argument('--use-gpt-agent', action='store_true', help='Use GPT as agent instead of SLM', default=False)
    parser.add_argument('--n-ex-subset', type=int, default=100, help='Number of examples to use in test split')
    parser.add_argument('--model', type=str, choices=['phi-3.5', 'qwen-7b'], default='phi-3.5')
    parser.add_argument('--use-single-screenshot-verifier', action='store_true', help='use last screenshots for verifier', default=True)
    parser.add_argument('--temp-summ-verf', type=float, default=0.01, help='temperature for the summarizer and verifier agents')

    # gpt4 args
    parser.add_argument('--use-spiral', action='store_true', help='Use regular openai a/c instead of azure')
    
    args = parser.parse_args()
    print(args)

    # iterate over all json files
    
    traj_dirs = json.load(open(args.input_file, 'r'))
    print('len(traj_dirs) = {}'.format(len(traj_dirs)))
          
    cnt = 0

    eval_data = []
    for i, folder in enumerate(traj_dirs):
        if i % 1000 == 0:
            print(i)
        js_file = os.path.join(folder, 'task_trajectory_data.json')
        if not os.path.exists(js_file):
            continue
        with open(js_file, 'r') as f:
            data = json.load(f)
        try:
            if len(data)==0:
                continue
            is_success = 'success' in data['verifier_agent_response'].split('\nStatus: ')[-1]
            if is_success:
                tmp = {}
                # print(data.keys())
                tmp[args.task_field] = data[args.task_field]
                tmp['init_url'] = data['init_url']
                tmp['log_dir'] = folder

                # print(tmp[args.task_field])

                # if 'regex' not in tmp[args.task_field]:
                eval_data.append(tmp)
                cnt += 1

                # if cnt >= args.n_ex_subset - 1:
                    # break

        except:
            traceback.print_exc()
            continue

    print('len(eval_data) = {}'.format(len(eval_data)))

    # sys.exit(0)

    # Create the dataset
    eval_dataset = create_huggingface_dataset(eval_data)

    # initialize the browser environment agent
    # browser_env_agent = ScriptBrowserEnvAgent(args, viewport_size={'width': args.viewport_width, 'height': args.viewport_height})

    flow = WebRandomWalkerFusionFlow(args)

    acc = 0.0
    total = 0

    for ex_id, example in enumerate(tqdm(eval_dataset)):
        log_dir = os.path.join(args.log_dir, example['log_dir'])

        print(log_dir)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        setup_logging(log_dir)

        logging.info(example)

        try:
            is_success = flow.run_task(example, log_dir)
        except:
            logging.info(traceback.format_exc())
            is_success = 0
            
        print('is_success = {}'.format(is_success))
        acc += is_success
        total += 1

        print('acc = {}'.format(acc))
        print('total = {}'.format(total))

        if ex_id >= args.n_ex_subset - 1:
            break
    
    print('Final accuracy = {}'.format(acc / total))

            


            
