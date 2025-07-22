from datasets import Dataset
import re
import random
import torch
import os
from PIL import Image
from tqdm import tqdm
import traceback

SYSTEM_MESSAGE = """You are an expert at completing instructions on Webpage screens. 
               You will be presented with a screenshot image with some numeric tags.
               If you decide to click somewhere, you should choose the numeric element idx that is the closest to the location you want to click.  
               You should decide the action to continue this instruction.
               You will be given the accessibility tree of the current screen in the format: '[element_idx] [role] [alt text or button name]'.
               Here are the available actions:
{"action": "goto", "action_natural_language": str, "value": <the url to go to>}
{"action": "click", "action_natural_language": str, "idx": <element_idx>}
{"action": "type", "action_natural_language": str, "idx": <element_idx>, "value": <the text to enter>}
{"action": "select", "action_natural_language": str, "idx": <element_idx>, "value": <the option to select>}
{"action": "scroll [up]", "action_natural_language": str}
{"action": "scroll [down]", "action_natural_language": str}
Your final answer must be in the above format.
"""
SYSTEM_MESSAGE_GS = """You are an expert at completing instructions on Webpage screens. 
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
"""
SYSTEM_MESSAGE_NEW = """You are an expert at completing instructions on Webpage screens. 
               You will be given a screenshot, in which interactive elements are outlined in bounding boxes of different colors. Each bounding box has a numeric ID label in the same color.
               If you decide to click somewhere, you should choose the numeric element idx that is the closest to the location you want to click.  
               
               Additionally, you will be given the accessibility tree of the current screen in the format: '[element_idx] [element type/role] [element text or button name] [action possible for this element]'.
               
               You should decide the action to continue the given instruction on a website.
               
               Here are the available actions:
{"action": "goto", "action_natural_language": str, "value": <the url to go to>}
{"action": "click", "action_natural_language": str, "idx": <element_idx>}
{"action": "type", "action_natural_language": str, "idx": <element_idx>, "value": <the text to enter>}
{"action": "select", "action_natural_language": str, "idx": <element_idx>, "value": <the option to select>}
{"action": "scroll [up]", "action_natural_language": str}
{"action": "scroll [down]", "action_natural_language": str}
Your final answer must be in the above format.
"""
USER_MESSAGE = """Here is the screenshot image: <|image_1|>\n
      The instruction is to {}. 
      History actions:
      {}\n\n
      Here is the screen information:
      {}\n\n
      Think about what you need to do with current screen, and output the action in the required format in the end. """

USER_MESSAGE_QWEN = """The instruction is to {}. 
      History actions:
      {}\n\n
      Here is the screen information:
      {}\n\n
      Think about what you need to do with current screen, and output the action in the required format in the end. """


def create_stepwise_dataset_phi(args, root, dataset, processor):
    train_data_stepwise = []

    for i in tqdm(range(len(dataset))):
        example = dataset[i]

        overall_task = example["summary_abstract"]

        if args.use_google_search:
            system_message = SYSTEM_MESSAGE_GS
        else:
            system_message = SYSTEM_MESSAGE

        system_message = {
            "role": "system",
            "content": system_message,
        }

        for sampled_step_id in range(len(example["actions"])):
            action_history = ""
            if sampled_step_id > 0:
                action_history = [
                    example["actions"][step_id]["step_action_nl"]
                    for step_id in range(sampled_step_id)
                ]
                action_history = "\n".join(action_history)

            if args.use_new_format:
                if example["actions"][sampled_step_id]["acc_tree_visible_before"]:
                    acc_tree = example["actions"][sampled_step_id][
                        "acc_tree_visible_before"
                    ]

                    if (
                        "acc_tree_other_before" in example["actions"][sampled_step_id]
                        and example["actions"][sampled_step_id]["acc_tree_other_before"]
                        and len(
                            example["actions"][sampled_step_id]["acc_tree_other_before"]
                        )
                        > 0
                    ):
                        acc_tree = (
                            acc_tree
                            + example["actions"][sampled_step_id][
                                "acc_tree_other_before"
                            ]
                        )
                else:
                    acc_tree = example["actions"][sampled_step_id]["acc_tree_before"]

                try:
                    if acc_tree and args.max_len_acctree > 0:
                        acc_tree = acc_tree[: args.max_len_acctree]
                    else:
                        acc_tree = ""
                except:
                    print(example["actions"][sampled_step_id])
            else:
                acc_tree = example["actions"][sampled_step_id]["acc_tree_before"][:4096]

            prompt_message = {
                "role": "user",
                "content": USER_MESSAGE.format(overall_task, action_history, acc_tree),
            }

            if args.use_new_format:
                image_path = os.path.join(
                    root,
                    example["folder"],
                    f"screenshot_som_crop_{sampled_step_id}.png",
                )
            else:
                image_path = os.path.join(
                    root, example["folder"], f"screenshot_som_{sampled_step_id}.png"
                )

            if not os.path.exists(image_path):
                image_path = os.path.join(
                    example["folder"], f"screenshot_som_{sampled_step_id}.png"
                )

            sample = {
                "system_message": system_message,
                "prompt_message": prompt_message,
                "image_path": image_path,
                "actions": example["actions"][sampled_step_id],
            }
            train_data_stepwise.append(sample)

    print("len(train_data_stepwise):", len(train_data_stepwise))
    dataset = Dataset.from_list(train_data_stepwise)
    return dataset


def create_stepwise_dataset_qwen(args, root, dataset, processor):
    train_data_stepwise = []

    for i in tqdm(range(len(dataset))):
        example = dataset[i]

        overall_task = example["summary_abstract"]

        if args.use_new_format:
            system_message = SYSTEM_MESSAGE_NEW
        elif args.use_google_search:
            system_message = SYSTEM_MESSAGE_GS
        else:
            system_message = SYSTEM_MESSAGE

        system_message = {
            "role": "system",
            "content": system_message,
        }

        for sampled_step_id in range(len(example["actions"])):
            action_history = ""
            if sampled_step_id > 0:
                action_history = [
                    example["actions"][step_id]["step_action_nl"]
                    for step_id in range(sampled_step_id)
                ]
                action_history = "\n".join(action_history)

            if args.use_new_format:
                if example["actions"][sampled_step_id]["acc_tree_visible_before"]:
                    acc_tree = example["actions"][sampled_step_id][
                        "acc_tree_visible_before"
                    ]

                    if (
                        "acc_tree_other_before" in example["actions"][sampled_step_id]
                        and example["actions"][sampled_step_id]["acc_tree_other_before"]
                        and len(
                            example["actions"][sampled_step_id]["acc_tree_other_before"]
                        )
                        > 0
                    ):
                        acc_tree = (
                            acc_tree
                            + example["actions"][sampled_step_id][
                                "acc_tree_other_before"
                            ]
                        )
                else:
                    acc_tree = example["actions"][sampled_step_id]["acc_tree_before"]

                try:
                    if acc_tree and args.max_len_acctree > 0:
                        acc_tree = acc_tree[: args.max_len_acctree]
                    else:
                        acc_tree = ""
                except:
                    print(example["actions"][sampled_step_id])
            else:
                acc_tree = example["actions"][sampled_step_id]["acc_tree_before"][:4096]

            prompt_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is the screenshot image:"},
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": USER_MESSAGE.format(
                            overall_task, action_history, acc_tree
                        ),
                    },
                ],
            }

            if args.use_new_format:
                image_path = os.path.join(
                    root,
                    example["folder"],
                    f"screenshot_som_crop_{sampled_step_id}.png",
                )
            else:
                image_path = os.path.join(
                    root, example["folder"], f"screenshot_som_{sampled_step_id}.png"
                )

            if not os.path.exists(image_path):
                image_path = os.path.join(
                    example["folder"], f"screenshot_som_{sampled_step_id}.png"
                )

            sample = {
                "system_message": system_message,
                "prompt_message": prompt_message,
                "image_path": image_path,
                "actions": example["actions"][sampled_step_id],
            }
            train_data_stepwise.append(sample)

    print("len(train_data_stepwise):", len(train_data_stepwise))
    dataset = Dataset.from_list(train_data_stepwise)
    return dataset


class WebTrajDataOrderedCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, data):
        image_path = data[0]["image_path"]
        system_message = data[0]["system_message"]
        prompt_message = data[0]["prompt_message"]
        actions = data[0]["actions"]

        image = Image.open(image_path)

        prompt = self.processor.tokenizer.apply_chat_template(
            [system_message, prompt_message], tokenize=False, add_generation_prompt=True
        )

        batch = self.processor(prompt, [image], return_tensors="pt")

        input_ids = [batch["input_ids"]]
        labels = [torch.tensor([-100] * len(batch["input_ids"][0])).unsqueeze(0)]

        answer_grounded = actions["new_action_grounded"]

        if not answer_grounded:
            answer_grounded = actions["action_grounded"]

        answer_type = answer_grounded.strip().split(" ")[0]
        answer_nl = actions["step_action_nl"]

        try:
            if answer_type == "click":
                match = re.search(r"click ?\[(\d+)\]", answer_grounded)
                element_id = match.group(1)
                answer = {
                    "action": answer_type,
                    "action_natural_language": answer_nl,
                    "idx": element_id,
                }
            elif answer_type == "type":
                match = re.search(r"type ?\[(\d+)\] ?\[(.+)\]", answer_grounded)
                element_id, text = (
                    match.group(1),
                    match.group(2),
                )
                answer = {
                    "action": answer_type,
                    "action_natural_language": answer_nl,
                    "idx": element_id,
                    "value": text,
                }
            elif answer_type == "select":
                match = re.search(r"select ?\[(\d+)\] ?\[(.+)\]", answer_grounded)
                element_id, text = (
                    match.group(1),
                    match.group(2),
                )
                answer = {
                    "action": answer_type,
                    "action_natural_language": answer_nl,
                    "idx": element_id,
                    "value": text,
                }
            elif answer_type == "scroll":
                match = re.search(r"scroll ?\[?(up|down)\]?", answer_grounded)
                direction = match.group(1)
                answer = {
                    "action": answer_type,
                    "action_natural_language": answer_nl,
                    "idx": direction,
                }
            elif answer_type == "goto":
                match = re.search(r"goto ?\[(.+)\]", answer_grounded)
                url = match.group(1)
                answer = {
                    "action": answer_type,
                    "action_natural_language": answer_nl,
                    "value": url,
                }
            elif answer_type == "google_search":
                answer_grounded = answer_grounded.replace("\n", "")
                match = re.search(r"google_search ?\[(.+)\]", answer_grounded)
                query = match.group(1)
                answer = {
                    "action": answer_type,
                    "action_natural_language": answer_nl,
                    "value": query,
                }
            else:
                answer = {
                    "action": answer_type,
                    "action_natural_language": answer_nl,
                    "idx": "",
                }
        except:
            answer = {
                "action": answer_type,
                "action_natural_language": answer_nl,
                "idx": "",
            }
            print(
                "exception: answer = {}, answer_grounded = {}".format(
                    answer, answer_grounded
                )
            )
            traceback.print_exc()

        answer = f"{answer}<|end|>\n<|endoftext|>"
        answer_input_ids = self.processor.tokenizer(
            answer, add_special_tokens=False, return_tensors="pt"
        )["input_ids"]
        input_ids.append(answer_input_ids)
        labels.append(answer_input_ids)
        assert "pixel_values" in batch, f"Image not found: {image_path}!!!\n"

        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        pixel_values = batch["pixel_values"]
        image_sizes = batch["image_sizes"]

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
        }

        return batch


class WebTrajDataOrderedQwenCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, data):
        image_path = data[0]["image_path"]
        system_message = data[0]["system_message"]
        prompt_message = data[0]["prompt_message"]
        actions = data[0]["actions"]

        image = Image.open(image_path)

        width, height = image.size
        max_width = 2168

        if width > max_width:
            crop_box = (0, 0, max_width, height)
            image = image.crop(crop_box)
            print("cropped to {}".format(image.size))

        messages = [system_message, prompt_message]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        batch = self.processor(
            text=[prompt], images=[image], padding=True, return_tensors="pt"
        )

        input_ids = [batch["input_ids"]]
        image_grid_thw = batch["image_grid_thw"]
        labels = [torch.tensor([-100] * len(batch["input_ids"][0])).unsqueeze(0)]

        answer_grounded = actions["new_action_grounded"]

        if not answer_grounded:
            answer_grounded = actions["action_grounded"]

        answer_type = answer_grounded.strip().split(" ")[0]
        answer_nl = actions["step_action_nl"]

        try:
            if answer_type == "click":
                match = re.search(r"click ?\[(\d+)\]", answer_grounded)
                element_id = match.group(1)
                answer = {
                    "action": answer_type,
                    "action_natural_language": answer_nl,
                    "idx": element_id,
                }
            elif answer_type == "type":
                match = re.search(r"type ?\[(\d+)\] ?\[(.+)\]", answer_grounded)
                element_id, text = (
                    match.group(1),
                    match.group(2),
                )
                answer = {
                    "action": answer_type,
                    "action_natural_language": answer_nl,
                    "idx": element_id,
                    "value": text,
                }
            elif answer_type == "select":
                match = re.search(r"select ?\[(\d+)\] ?\[(.+)\]", answer_grounded)
                element_id, text = (
                    match.group(1),
                    match.group(2),
                )
                answer = {
                    "action": answer_type,
                    "action_natural_language": answer_nl,
                    "idx": element_id,
                    "value": text,
                }
            elif answer_type == "scroll":
                match = re.search(r"scroll ?\[?(up|down)\]?", answer_grounded)
                direction = match.group(1)
                answer = {
                    "action": answer_type,
                    "action_natural_language": answer_nl,
                    "idx": direction,
                }
            elif answer_type == "goto":
                match = re.search(r"goto ?\[(.+)\]", answer_grounded)
                url = match.group(1)
                answer = {
                    "action": answer_type,
                    "action_natural_language": answer_nl,
                    "value": url,
                }
            elif answer_type == "google_search":
                answer_grounded = answer_grounded.replace("\n", "")
                match = re.search(r"google_search ?\[(.+)\]", answer_grounded)
                query = match.group(1)
                answer = {
                    "action": answer_type,
                    "action_natural_language": answer_nl,
                    "value": query,
                }
            else:
                answer = {
                    "action": answer_type,
                    "action_natural_language": answer_nl,
                    "idx": "",
                }
        except:
            answer = {
                "action": answer_type,
                "action_natural_language": answer_nl,
                "idx": "",
            }
            print(
                "exception: answer = {}, answer_grounded = {}".format(
                    answer, answer_grounded
                )
            )
            traceback.print_exc()

        answer = f"{answer}<|im_end|>\n<|endoftext|>"
        answer_input_ids = self.processor.tokenizer(
            answer, add_special_tokens=False, return_tensors="pt"
        )["input_ids"]
        input_ids.append(answer_input_ids)
        labels.append(answer_input_ids)
        assert "pixel_values" in batch, f"Image not found: {image_path}!!!\n"

        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        pixel_values = batch["pixel_values"]

        attention_mask = torch.ones_like(input_ids)

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "attention_mask": attention_mask,
        }

        return batch
