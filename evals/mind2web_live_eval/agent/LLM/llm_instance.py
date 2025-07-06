from .openai import GPTGenerator, GPTGeneratorWithJSON
from .claude import ClaudeGenerator
from .gemini import GeminiGenerator
from .togetherai import TogetherAIGenerator
from .hf_models import Phi3Generator, Phi3MiniGenerator, Mistral7BInstructGenerator, LlavaNextGenerator, Qwen2VLGenerator, InternVLGenerator
from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
import torch
from peft import PeftModel
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, Qwen2VLForConditionalGeneration
import logging
from accelerate import load_checkpoint_and_dispatch

logger = logging.getLogger(__name__)



def create_model(model_name_or_path, model_name, use_flash_attention=False):
    bnb_config = None

    # for full finetuning, GPU memory can't be cleared (likely caused by deepspeed
    # https://github.com/microsoft/DeepSpeed/issues/3677)
    # so we don't reload the model

    logging.info(f"Loading model from {model_name_or_path}")
    logging.info('use_flash_attention: {}'.format(use_flash_attention))
    logging.info('bnb_config: {}'.format(bnb_config))

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        # Phi-3-V is originally trained in bf16 + flash attn
        # For fp16 mixed precision training, load in f32 to avoid hf accelerate error
        torch_dtype=torch.bfloat16 if use_flash_attention else torch.float32,
        trust_remote_code=True,
        _attn_implementation='flash_attention_2' if use_flash_attention else 'eager',
        quantization_config=bnb_config,
    )

    model.eval()

    return model

def create_llm_instance(args, model, json_mode=False, all_json_models=None):
    
    logger.info(f"Creating LLM instance for model: {model}")

    if "gpt" in model:
        if json_mode:
            if model in all_json_models:
                return GPTGeneratorWithJSON(model)
            else:
                raise ValueError("The text model does not support JSON mode.")
        else:
            return GPTGenerator(args, model)
    elif "claude" in model:
        if json_mode:
            raise ValueError("Claude does not support JSON mode.")
        else:
            return ClaudeGenerator(model)
    elif "gemini" in model:
        if json_mode:
            raise ValueError("Gemini does not support JSON mode.")
        else:
            return GeminiGenerator(model)
    elif model == "phi-3.5v":
        model_id = 'microsoft/Phi-3.5-vision-instruct'

        if len(args.ckpt_path) > 0:
            model_name_or_path = args.ckpt_path
        else:
            model_name_or_path = 'microsoft/Phi-3.5-vision-instruct'

        model = create_model(model_name_or_path, model_id, use_flash_attention=args.use_flash_attention, use_qlora=args.use_qlora)
        model.to('cuda')
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=args.num_crops)

        return Phi3Generator(model, processor, add_repetition_penalty=args.add_repetition_penalty, repetition_penalty=args.repetition_penalty, temperature=args.temp, do_sample=not args.use_greedy)

    elif model == "phi-3v":
        model_id = "microsoft/Phi-3-vision-128k-instruct"

        if len(args.ckpt_path) > 0:
            model_name_or_path = args.ckpt_path
        else:
            model_name_or_path = 'microsoft/Phi-3-vision-128k-instruct'

        model = create_model(model_name_or_path, model_id, use_flash_attention=args.use_flash_attention, use_qlora=args.use_qlora)
        model.to('cuda')
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=args.num_crops)

        return Phi3Generator(model, processor, add_repetition_penalty=args.add_repetition_penalty, repetition_penalty=args.repetition_penalty, temperature=args.temp, do_sample=not args.use_greedy)
    
    elif model == "phi3mini":
        model_id = 'microsoft/Phi-3.5-mini-instruct'

        if len(args.ckpt_path) > 0:
            model_name_or_path = args.ckpt_path
        else:
            model_name_or_path = 'microsoft/Phi-3.5-mini-instruct'

        model = create_model(model_name_or_path, model_id, use_flash_attention=args.use_flash_attention, use_qlora=args.use_qlora)
        model.to('cuda')
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=args.num_crops)

        return Phi3MiniGenerator(model, processor, add_repetition_penalty=args.add_repetition_penalty, repetition_penalty=args.repetition_penalty, temperature=args.temp, do_sample=not args.use_greedy)

    elif model == "Mistral-7B-Instruct":
        model_id = "mistralai/Mistral-7B-Instruct-v0.3"

        if len(args.ckpt_path) > 0:
            model_name_or_path = args.ckpt_path
        else:
            model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.3"

        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
        model.to('cuda')
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        return Mistral7BInstructGenerator(model, tokenizer, add_repetition_penalty=args.add_repetition_penalty, repetition_penalty=args.repetition_penalty, temperature=args.temp, do_sample=not args.use_greedy)

    elif model == "llava-v1.6-mistral-7b":
        if len(args.ckpt_path) > 0:
            model_name_or_path = args.ckpt_path
        else:
            model_name_or_path = "llava-hf/llava-v1.6-mistral-7b-hf"

        model = LlavaNextForConditionalGeneration.from_pretrained(model_name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True) 
        model.to("cuda:0")

        processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

        return LlavaNextGenerator(model, processor, temperature=args.temp, do_sample=not args.use_greedy)

    elif model == "llava-v1.6-vicuna-13b-hf":
        if len(args.ckpt_path) > 0:
            model_name_or_path = args.ckpt_path
        else:
            model_name_or_path = "llava-hf/llava-v1.6-vicuna-13b-hf"

        model = LlavaNextForConditionalGeneration.from_pretrained(model_name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True) 
        model.to("cuda:0")

        processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

        return LlavaNextGenerator(model, processor, temperature=args.temp, do_sample=not args.use_greedy)
    
    elif model == 'qwen2-vl-7b':
        logger.info(f"inside qwen2-vl-7b")

        # model_id = "Qwen/Qwen2-VL-7B-Instruct"
        if len(args.ckpt_path) > 0:
            model_name_or_path = args.ckpt_path
        else:
            model_name_or_path = "Qwen/Qwen2-VL-7B-Instruct"

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        )
        # if len(args.ckpt_path) > 0:
        #     model = load_checkpoint_and_dispatch(
        #         model, checkpoint=args.ckpt_path, device_map="auto"
        #     )

        model.to("cuda:0")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        return Qwen2VLGenerator(model, processor, temperature=args.temp, do_sample=not args.use_greedy)
    
    elif model == 'qwen2-vl-72b':
        # logger.info(f"inside qwen2-vl-7b")
        # model_id = "Qwen/Qwen2-VL-72B-Instruct"

        if len(args.ckpt_path) > 0:
            model_name_or_path = args.ckpt_path
        else:
            model_name_or_path = "Qwen/Qwen2-VL-72B-Instruct"

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            # model_id,
            torch_dtype="auto",
            # attn_implementation="flash_attention_2",
            device_map="auto"
        )
        # if len(args.ckpt_path) > 0:
        #     model = load_checkpoint_and_dispatch(
        #         model, checkpoint=args.ckpt_path, device_map="auto"
        #     )

        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-72B-Instruct")

        return Qwen2VLGenerator(model, processor, temperature=args.temp, do_sample=not args.use_greedy)
    
    elif model == 'intern-vl-8b':

        if len(args.ckpt_path) > 0:
            model_name_or_path = args.ckpt_path
        else:
            model_name_or_path = "OpenGVLab/InternVL2-8B"

        model = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True).eval().cuda()
    
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False)
        return InternVLGenerator(model, tokenizer, temperature=args.temp, do_sample=not args.use_greedy)

    else:
        if json_mode:
            raise ValueError("TogetherAI does not support JSON mode.")
        else:
            return TogetherAIGenerator(model)

async def semantic_match_llm_request(messages: list = None, args=None):
    GPT35 = GPTGenerator(model="gpt-3.5-turbo", args=args)
    return await GPT35.request(messages)

async def semantic_match_llm_request_sync(messages: list = None, args=None):
    GPT35 = GPTGenerator(model="gpt-3.5-turbo", args=args)
    return GPT35.request_sync(messages)