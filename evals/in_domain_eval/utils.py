'''
Credit: https://github.com/web-arena-x/visualwebarena/
'''

import aiohttp
import json
import traceback
import requests
import time
from PIL import Image
import logging
import os
import base64
from openai import AzureOpenAI
from azure.identity import AzureCliCredential, DefaultAzureCredential, get_bearer_token_provider
from azure.identity import  ManagedIdentityCredential

import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, TypedDict, Union

import numpy as np
import numpy.typing as npt
from beartype import beartype
from PIL import Image
import nltk
import random
import os
from multiprocessing import Pool
from nltk.translate.bleu_score import SmoothingFunction
import tiktoken
import re
from transformers import AutoModelForCausalLM
import torch

class CredentialException(Exception):
    pass

def call_gpt4v(args, messages, max_tokens=2048, temperature=0.01):
    
    # endpoint = "https://dataoai2.openai.azure.com/"
    # endpoint = "https://yadaoai.openai.azure.com/"
    
    # deployment = "dataoai2-gpt4"
    # deployment = 'gpt-4o'

    # deployment = 'gpt4o_2' 
    # api_version="2024-05-01-preview"

    if args.api_auth_type == 'azurecli':
        token_provider = get_bearer_token_provider(AzureCliCredential(), "https://cognitiveservices.azure.com/.default")
    else:
        scope = "https://cognitiveservices.azure.com/.default"
        try:
            credential = ManagedIdentityCredential()
            # Check if given credential can get token successfully.
            token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
            token = token_provider()
        except Exception as exception:
            # log something here
            raise CredentialException(f"Could not get a token for scope={scope}:\n{traceback.format_exc()}")
    
    client = AzureOpenAI(
        azure_endpoint=args.endpoint,
        azure_ad_token_provider=token_provider,
        api_version=args.api_version,
    )
    max_num_trial = 3
    num_trial = 0
    call_api_success = True

    while num_trial < max_num_trial:
        try:
            completion = client.chat.completions.create(model=args.deployment, messages=messages, temperature=temperature, max_tokens=max_tokens)
            a = json.loads(completion.json())
            ans_1st_pass = a['choices'][0]['message']['content']
            break
        except:
            logging.info('retry call gptv {}'.format(num_trial))
            num_trial += 1
            ans_1st_pass = ''
            time.sleep(10)
    
    if num_trial == max_num_trial:
        call_api_success = False

    return ans_1st_pass, call_api_success

def create_model(model_name_or_path, use_flash_attention=False, use_qlora=False):
    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16 if use_flash_attention else torch.float16,
        )
        if use_qlora
        else None
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        # Phi-3-V is originally trained in bf16 + flash attn
        # For fp16 mixed precision training, load in f32 to avoid hf accelerate error
        torch_dtype=torch.bfloat16 if use_flash_attention else torch.float32,
        trust_remote_code=True,
        _attn_implementation='flash_attention_2' if use_flash_attention else 'eager',
        quantization_config=bnb_config,
    )

    return model

@dataclass
class DetachedPage:
    url: str
    content: str  # html


@beartype
def png_bytes_to_numpy(png: bytes) -> npt.NDArray[np.uint8]:
    """Convert png bytes to numpy array

    Example:

    >>> fig = go.Figure(go.Scatter(x=[1], y=[1]))
    >>> plt.imshow(png_bytes_to_numpy(fig.to_image('png')))
    """
    return np.array(Image.open(BytesIO(png)))


def pil_to_b64(img: Image.Image) -> str:
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
        img_b64 = base64.b64encode(byte_data).decode("utf-8")
        img_b64 = "data:image/png;base64," + img_b64
    return img_b64


def pil_to_vertex(img: Image.Image) -> str:
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
        img_vertex = VertexImage.from_bytes(byte_data)
    return img_vertex


class AccessibilityTreeNode(TypedDict):
    nodeId: str
    ignored: bool
    role: dict[str, Any]
    chromeRole: dict[str, Any]
    name: dict[str, Any]
    properties: list[dict[str, Any]]
    childIds: list[str]
    parentId: str
    backendDOMNodeId: int
    frameId: str
    bound: list[float] | None
    union_bound: list[float] | None
    offsetrect_bound: list[float] | None


class BrowserConfig(TypedDict):
    win_upper_bound: float
    win_left_bound: float
    win_width: float
    win_height: float
    win_right_bound: float
    win_lower_bound: float
    device_pixel_ratio: float


class BrowserInfo(TypedDict):
    DOMTree: dict[str, Any]
    config: BrowserConfig


AccessibilityTree = list[AccessibilityTreeNode]


Observation = str | npt.NDArray[np.uint8]


class StateInfo(TypedDict):
    observation: dict[str, Observation]
    info: Dict[str, Any]

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
import time
import logging
import os

def setup_logging(ex_log_dir):
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create a new file handler
    log_file = os.path.join(ex_log_dir, 'step_simulator_flow.log')
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w', format='%(asctime)s - %(message)s')

def call_gpt4v_new(message_text, image_path=None, max_tokens=2048):
    if image_path:
        try:
            with open(image_path, "rb") as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode('ascii')
        except: 
            encoded_image = image_path
    
    if image_path:
        content = [{"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}, {"type": "text","text": message_text},]
    else:
        content = [{"type": "text","text": message_text},]

    max_num_trial = 3
    num_trial = 0
    call_api_success = True

    endpoint = "https://yadaoai.openai.azure.com/"
    # deployment = "dataoai2-gpt4"
    deployment = 'gpt-4o'
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version="2024-02-01",
    )
    while num_trial < max_num_trial:
        try:
            response = client.chat.completions.create(
                            model=deployment,
                            temperature=0.01,
                            messages=[
                                        {
                                        "role": "system",
                                        "content": [
                                            {
                                            "type": "text",
                                            "text": "You are an AI assistant that is good at making plans and analyzing screens, and helping people find information."
                                            },
                                        ]
                                        },
                                        {
                                        "role": "user",
                                        "content": content
                                        }
                                    ],
                        )
            ans_1st_pass = response.choices[0].message.content
            break
        except:
            print('retry call gptv', num_trial)
            num_trial += 1
            ans_1st_pass = ''
            time.sleep(10)
    if num_trial == max_num_trial:
        call_api_success = False
    return ans_1st_pass, call_api_success

def get_reference(self):
    if self.reference is None:
        reference = list()
        with open(self.test_data) as real_data:
            for text in real_data:
                text = nltk.word_tokenize(text)
                reference.append(text)
        self.reference = reference
        return reference
    else:
        return self.reference

def get_bleu_fast(reference, sample_size):
    random.shuffle(reference)
    reference = reference[0:sample_size]
    return get_bleu_parallel(reference=reference)


def get_bleu_parallel(ngram=3, reference=None):
    if reference is None:
        reference = get_reference()
    weight = tuple((1. / ngram for _ in range(ngram)))
    pool = Pool(os.cpu_count())
    result = list()
    sentence_num = len(reference)
    for index in range(sentence_num):
        hypothesis = reference[index]
        other = reference[:index] + reference[index+1:]
        result.append(pool.apply_async(calc_bleu, args=(other, hypothesis, weight)))

    score = 0.0
    cnt = 0
    for i in result:
        score += i.get()
        cnt += 1
    pool.close()
    pool.join()
    return score / cnt

def calc_bleu(reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                smoothing_function=SmoothingFunction().method1)

def calc_num_tokens(messages):
    n_tokens = 0
    encoding = tiktoken.encoding_for_model("gpt-4o")

    for message in messages:
        items = message['content']

        for item in items:
            if item['type'] == 'text':
                n_tokens += len(encoding.encode(item['text']))
            elif item['type'] == 'image_url':
                n_tokens += 1100
        
    return n_tokens

def get_top_domain(input_string):
    # Split the input string using '/' or ':' as the separator
    fields = re.split(r'[/:]', input_string)
    if len(fields) < 4:
        raise ValueError("Input string does not have enough fields to extract the fourth one.")
    # Extract the fourth field
    fourth_field = fields[3]
    # Define the regex pattern and the replacement pattern for the sed operation
    pattern = r'([a-zA-Z0-9-]+)\.([a-zA-Z]{2,})$'
    replacement = r'\1'
    # Perform the substitution using re.sub
    transformed_string = re.sub(pattern, replacement, fourth_field)
    return transformed_string

