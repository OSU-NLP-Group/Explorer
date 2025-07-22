import argparse
from ast import parse
import json
import os
import random
from pathlib import Path

import Levenshtein
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import transformers
import traceback

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
    TrainerCallback
)

from PIL import Image
import re
from train_utils import create_stepwise_dataset_qwen, WebTrajDataOrderedQwenCollator

random.seed(123937)

# suggested deepspeed config
DS_CONFIG_DICT = {
    'zero_optimization': {
        'stage': 3,
        'allgather_partitions': True,
        'allgather_bucket_size': 5e8,
        'overlap_comm': True,
        'reduce_scatter': True,
        'reduce_bucket_size': 5e8,
        'contiguous_gradients': True,
        'round_robin_gradients': True,
        'stage3_gather_16bit_weights_on_model_save': True
    },
    'fp16': {
        'enabled': 'auto',
        'loss_scale': 0,
        'loss_scale_window': 1000,
        'initial_scale_power': 16,
        'hysteresis': 2,
        'min_loss_scale': 1,
    },
    'bf16': {'enabled': 'auto'},
    'train_micro_batch_size_per_gpu': 'auto',
    'train_batch_size': 'auto',
    'gradient_accumulation_steps': 'auto',
    'gradient_clipping': 'auto',
}


def create_model(model_name_or_path, use_flash_attention=False):

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )

    return model


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

SYSTEM_MESSAGE_NOGOTO_GS = '''You are an expert at completing instructions on Webpage screens. 
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

USER_MESSAGE = '''The instruction is to {}. 
      History actions:
      {}\n\n
      Here is the screen information:
      {}\n\n
      Think about what you need to do with current screen, and output the action in the required format in the end. '''


class WebTrajDataStepWiseCollator:
    def __init__(self, args, processor, root, max_steps=1):
        self.max_steps = max_steps
        self.processor = processor
        self.root = root
        self.args = args

    def __call__(self, data):
        assert len(data) == 1, f'Phi-3-V only supports batch_size == 1, got {len(data)}'
        data = data[0]
        overall_task = data['summary_abstract']

        if self.args.use_google_search:
            system_message = SYSTEM_MESSAGE_GS
        elif self.args.use_nogoto_gs_format:
            system_message = SYSTEM_MESSAGE_NOGOTO_GS
        else:
            return NotImplementedError

        system_message = {
            'role': 'system',
            'content': system_message,
        }
        
        # print(system_message)
        
        sampled_step_id = random.randint(0, len(data['actions']) - 1)
        action_history = ''
        if sampled_step_id > 0:    
            action_history = [data['actions'][step_id]['step_action_nl'] for step_id in range(sampled_step_id)]
            action_history = '\n'.join(action_history)

        if self.args.use_new_format:
            if data['actions'][sampled_step_id]['acc_tree_visible_before']:
                acc_tree = data['actions'][sampled_step_id]['acc_tree_visible_before']

                if 'acc_tree_other_before' in data['actions'][sampled_step_id] and data['actions'][sampled_step_id]['acc_tree_other_before'] and len(data['actions'][sampled_step_id]['acc_tree_other_before']) > 0:
                    acc_tree = acc_tree + data['actions'][sampled_step_id]['acc_tree_other_before']
            else:
                acc_tree = data['actions'][sampled_step_id]['acc_tree_before']
            
            try:
                if acc_tree and self.args.max_len_acctree > 0:
                    acc_tree = acc_tree[:self.args.max_len_acctree]
                else:
                    acc_tree = ''
            except:
                print(data['actions'][sampled_step_id])
        else:
            acc_tree = data['actions'][sampled_step_id]['acc_tree_before'][:4096]

        if self.args.use_new_format:
            image_path = os.path.join(self.root, data['folder'], f'screenshot_som_crop_{sampled_step_id}.png')
        else:
            image_path = os.path.join(self.root, data['folder'], f'screenshot_som_{sampled_step_id}.png')
        
        if not os.path.exists(image_path):
            image_path = os.path.join(data['folder'], f'screenshot_som_{sampled_step_id}.png')
            
        prompt_message = {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'Here is the screenshot image:'},
                {"type": "image"},
                {'type': 'text', 'text': USER_MESSAGE.format(overall_task, action_history, acc_tree)}
            ]
        }
        
        image = Image.open(image_path)

        width, height = image.size
        max_width = 2168

        if width > max_width:
            crop_box = (0, 0, max_width, height)
            image = image.crop(crop_box)
            # print('cropped to {}'.format(image.size))
        
        messages = [system_message, prompt_message]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # print('prompt = {}'.format(prompt))
        
        batch = self.processor(text=[prompt], images=[image], padding=True, return_tensors="pt")

        input_ids = [batch['input_ids']]
        labels = [torch.tensor([-100] * len(batch['input_ids'][0])).unsqueeze(0)]
        image_grid_thw = batch['image_grid_thw']
        
        answer_grounded = data['actions'][sampled_step_id]['new_action_grounded']

        if not answer_grounded:
            answer_grounded = data['actions'][sampled_step_id]['action_grounded']

        answer_type = answer_grounded.strip().split(' ')[0]
        answer_nl = data['actions'][sampled_step_id]['step_action_nl']
            
        try:
            if answer_type == 'click':
                match = re.search(r"click ?\[(\d+)\]", answer_grounded)
                element_id = match.group(1)
                answer = {"action": answer_type, "action_natural_language": answer_nl, "idx": element_id}
            elif answer_type == 'type':
                match = re.search(
                    r"type ?\[(\d+)\] ?\[(.+)\]", answer_grounded
                )
                element_id, text = (
                    match.group(1),
                    match.group(2),
                )
                answer = {"action": answer_type, "action_natural_language": answer_nl, "idx": element_id, "value": text}
            elif answer_type == 'select':
                match = re.search(
                    r"select ?\[(\d+)\] ?\[(.+)\]", answer_grounded
                )
                element_id, text = (
                    match.group(1),
                    match.group(2),
                )
                answer = {"action": answer_type, "action_natural_language": answer_nl, "idx": element_id, "value": text}
            elif answer_type == 'scroll':
                match = re.search(r"scroll ?\[?(up|down)\]?", answer_grounded)
                direction = match.group(1)
                answer = {"action": answer_type, "action_natural_language": answer_nl, "idx": direction}
            elif answer_type == 'goto':
                match = re.search(r"goto ?\[(.+)\]", answer_grounded)
                url = match.group(1)
                answer = {"action": answer_type, "action_natural_language": answer_nl, "value": url}
            elif answer_type == 'google_search':
                answer_grounded = answer_grounded.replace('\n', '')
                match = re.search(r"google_search ?\[(.+)\]", answer_grounded)
                query = match.group(1)
                answer = {"action": answer_type, "action_natural_language": answer_nl, "value": query}
            else:
                answer = {"action": answer_type, "action_natural_language": answer_nl, "idx": ""}
        except:
            answer = {"action": answer_type, "action_natural_language": answer_nl, "idx": ""}
            print('exception: answer = {}, answer_grounded = {}'.format(answer, answer_grounded))
            print('actions = {}'.format(data['actions']))
            traceback.print_exc()

        # print('answer = {}'.format(answer))
        # print('answer_grounded = {}, answer = {}'.format(answer_grounded, answer))

        answer = f'{answer}<|im_end|>\n<|endoftext|>'
        answer_input_ids = self.processor.tokenizer(answer, add_special_tokens=False, return_tensors='pt')['input_ids']
        input_ids.append(answer_input_ids)
        labels.append(answer_input_ids)
        assert 'pixel_values' in batch, f'Image not found: {image_path}!!!\n'
        
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        pixel_values = batch['pixel_values']

        attention_mask = torch.ones_like(input_ids)

        batch = {'input_ids': input_ids, 'labels': labels, 'pixel_values': pixel_values, 'image_grid_thw': image_grid_thw, 'attention_mask': attention_mask}
        # print(batch)

        return batch


class SafeTrainer(Trainer):
    def training_step(self, model, inputs):
        try:
            # Run the standard training step
            return super().training_step(model, inputs)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("Out of memory error detected. Skipping this example.")
                # Clear the CUDA cache to recover memory
                torch.cuda.empty_cache()
                return torch.tensor(0.0, device=next(model.parameters()).device)
            else:
                raise e  # Re-raise if it's not an OOM error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default='Qwen/Qwen2-VL-7B-Instruct',
        help='Model name or path to load from',
    )
    parser.add_argument(
        '--full_train', action='store_true', help='Use full training dataset (DocVQA)'
    )
    parser.add_argument('--stage', type=int, help='DS train stage')
    parser.add_argument('--use_flash_attention', action='store_true', help='Use Flash Attention')
    parser.add_argument('--bf16', action='store_true', help='Use BF16')
    parser.add_argument('--train_dir', type=str, help='train data dir', default='')
    parser.add_argument('--train_dir_order', type=str, help='train data dir', default='')
    
    parser.add_argument('--train_data_dir', type=str, help='train data dir', default='')
    parser.add_argument('--output_dir', type=str, default='./output/', help='Output directory')
    parser.add_argument('--save-strategy', type=str, default='steps', help='Save strategy')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument(
        '--num_train_epochs', type=int, default=1, help='Number of training epochs'
    )
    parser.add_argument('--learning_rate', type=float, default=4.0e-5, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--no-tqdm', dest='tqdm', action='store_false', help='Disable tqdm')
    parser.add_argument('--tensorboard-logging', action='store_true', help='log to tensorboard')
    parser.add_argument('--order_all_steps', action='store_true', help='enforce all steps are seen in first epoch in same order')
    parser.add_argument('--report_to', type=str, default='wandb', help='report to')
    parser.add_argument('--use-new-format', action='store_true', help='use new format for traj data')
    parser.add_argument('--max-len-acctree', type=int, default=-1, help='max len acctree')
    parser.add_argument('--use-google-search', action='store_true', help='add google search in action space and prompt')
    parser.add_argument('--use-nogoto-gs-format', action='store_true', help='remove gs and goto from prompt')

    args = parser.parse_args()

    accelerator = Accelerator()

    with accelerator.local_main_process_first():
        processor = AutoProcessor.from_pretrained(
            args.model_name_or_path, trust_remote_code=True
        )
        model = create_model(
            args.model_name_or_path,
            use_flash_attention=args.use_flash_attention,
        )
        
    if args.order_all_steps:
        if not os.path.exists(args.train_dir_order):
            train_dataset = load_from_disk(args.train_dir)

            train_dataset = create_stepwise_dataset_qwen(args, args.train_data_dir, train_dataset, processor)
            train_dataset.save_to_disk(args.train_dir_order)
        else:
            train_dataset = load_from_disk(args.train_dir_order)
    else:
        train_dataset = load_from_disk(args.train_dir)

    print('train_dataset:', train_dataset)
    print('len(train_dataset):', len(train_dataset))
    
    import time
    time.sleep(3)

    num_gpus = accelerator.num_processes
    print(f'training on {num_gpus} GPUs')
    assert args.batch_size % num_gpus == 0, 'Batch size must be divisible by the number of GPUs'
    gradient_accumulation_steps = args.batch_size // num_gpus
    if args.bf16:
        fp16 = False
        bf16 = True
    else:
        fp16 = True
        bf16 = False

    # hard coded training args
    training_args = TrainingArguments(
        ddp_find_unused_parameters=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=1,  # NOTE currently only supports batch_size == 1
        per_device_eval_batch_size=1,
        gradient_checkpointing=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim='adamw_torch',
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-7,
        learning_rate=args.learning_rate,
        weight_decay=args.wd,
        max_grad_norm=1.0,
        lr_scheduler_type='linear',
        warmup_steps=50,
        logging_steps=10,
        output_dir=args.output_dir,
        save_strategy=args.save_strategy,
        save_steps=100,
        # save_steps=1,
        save_total_limit=1 if args.save_strategy=='steps' else None,
        save_only_model=True,
        bf16=bf16,
        fp16=fp16,
        remove_unused_columns=False,
        report_to='tensorboard' if args.tensorboard_logging else 'none',
        deepspeed=DS_CONFIG_DICT,
        disable_tqdm=not args.tqdm,
        dataloader_num_workers=1, # 4,
        dataloader_prefetch_factor=1, #2,
    )
    
    if args.order_all_steps:
        data_collator = WebTrajDataOrderedQwenCollator(processor)
    else:
        data_collator = WebTrajDataStepWiseCollator(args, processor, args.train_data_dir, args.model_name_or_path)
    
    # eval before fine-tuning
    out_path = Path(training_args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    model = model.to(f'cuda:{local_rank}')

    trainer = SafeTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    trainer.train()
    
    accelerator.wait_for_everyone()

if __name__ == '__main__':
    main()
