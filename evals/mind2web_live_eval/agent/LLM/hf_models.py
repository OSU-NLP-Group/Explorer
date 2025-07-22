import sys
import asyncio
from functools import partial
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from sanic.log import logger
from evals.mind2web_live_eval.agent.Utils import *
from .token_cal import truncate_messages_based_on_estimated_tokens
import traceback
from PIL import Image
from .utils_internvl import load_image


class Phi3Generator:
    def __init__(
        self,
        model,
        processor,
        add_repetition_penalty=False,
        repetition_penalty=1.1,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
    ):
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.add_repetition_penalty = add_repetition_penalty
        self.repetition_penalty = repetition_penalty
        self.do_sample = do_sample

    async def request(self, messages: list = None, image=None):
        try:
            cpu_count = multiprocessing.cpu_count()
            with ThreadPoolExecutor(max_workers=cpu_count * 2) as pool:
                future_answer = pool.submit(self.chat, messages, image)
                future_answer_result = await future_answer.result()
                model_response = future_answer_result  # 获取第一个选项
                return model_response, ""
        except Exception as e:
            logger.error(f"Error in Phi3Generator.request: {e}")
            logger.info(traceback.format_exc)
            return "", str(e)

    def request_sync(self, messages=None, image=None):
        return self.get_response(messages, image)

    async def chat(self, messages, image):
        loop = asyncio.get_event_loop()

        func = partial(self.get_response, messages=messages, image=image)

        return await loop.run_in_executor(None, func)

    def get_response(self, messages, image=None):
        # print(f"messages: {messages}")
        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        logger.info("image = {}".format(image))

        if image is not None:
            inputs = self.processor(prompt, [image], return_tensors="pt").to("cuda:0")
        else:
            # print('image is None')
            inputs = self.processor(prompt, return_tensors="pt").to("cuda:0")

        generation_args = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
        }

        if self.add_repetition_penalty:
            generation_args["repetition_penalty"] = self.repetition_penalty

        generate_ids = self.model.generate(
            **inputs,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **generation_args,
        )

        # remove input tokens
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return response


class Phi3MiniGenerator:
    def __init__(
        self,
        model,
        processor,
        add_repetition_penalty=False,
        repetition_penalty=1.1,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
    ):
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.add_repetition_penalty = add_repetition_penalty
        self.repetition_penalty = repetition_penalty
        self.do_sample = do_sample

    async def request(self, messages: list = None, image=None):
        try:
            cpu_count = multiprocessing.cpu_count()
            with ThreadPoolExecutor(max_workers=cpu_count * 2) as pool:
                future_answer = pool.submit(self.chat, messages, image)
                future_answer_result = await future_answer.result()
                model_response = future_answer_result  # 获取第一个选项
                return model_response, ""
        except Exception as e:
            logger.error(f"Error in Phi3Generator.request: {e}")
            logger.info(traceback.format_exc)
            return "", str(e)

    def request_sync(self, messages=None, image=None):
        return self.get_response(messages, image)

    async def chat(self, messages, image):
        loop = asyncio.get_event_loop()

        func = partial(self.get_response, messages=messages, image=image)

        return await loop.run_in_executor(None, func)

    def get_response(self, messages, image=None):
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(prompt, return_tensors="pt").to("cuda:0")

        generation_args = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
        }

        if self.add_repetition_penalty:
            generation_args["repetition_penalty"] = self.repetition_penalty

        generate_ids = self.model.generate(
            **inputs, eos_token_id=self.processor.eos_token_id, **generation_args
        )

        # remove input tokens
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return response


class Mistral7BInstructGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        add_repetition_penalty=False,
        repetition_penalty=1.1,
        max_new_tokens=500,
        temperature=0.7,
        do_sample=True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.add_repetition_penalty = add_repetition_penalty
        self.repetition_penalty = repetition_penalty
        self.do_sample = do_sample

    async def request(
        self,
        messages: list = None,
        image=None,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> (str, str):
        try:
            cpu_count = multiprocessing.cpu_count()
            with ThreadPoolExecutor(max_workers=cpu_count * 2) as pool:
                future_answer = pool.submit(self.chat, messages, image)
                future_answer_result = await future_answer.result()
                model_response = future_answer_result  # 获取第一个选项
                return model_response, ""
        except Exception as e:
            logger.error(f"Error in Mistral7BInstructGenerator.request: {e}")
            logger.info(traceback.format_exc)
            sys.exit(0)
            return "", str(e)

    def request_sync(self, messages=None, image=None):
        return self.get_response(messages, image)

    async def chat(self, messages, image):
        loop = asyncio.get_event_loop()

        func = partial(self.get_response, messages=messages, image=image)

        return await loop.run_in_executor(None, func)

    def get_response(self, messages, image=None):
        tool_use_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(tool_use_prompt, return_tensors="pt").to("cuda")

        generation_args = {
            "max_new_tokens": 512,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
        }

        if self.add_repetition_penalty:
            generation_args["repetition_penalty"] = self.repetition_penalty

        generate_ids = self.model.generate(**inputs, **generation_args)

        # remove input tokens
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]

        response = self.tokenizer.decode(generate_ids[0], skip_special_tokens=True)
        return response


class LlavaNextGenerator:
    def __init__(
        self, model, processor, max_new_tokens=512, temperature=0.7, do_sample=True
    ):
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

    async def request(self, messages: list = None, image=None):
        try:
            cpu_count = multiprocessing.cpu_count()
            with ThreadPoolExecutor(max_workers=cpu_count * 2) as pool:
                future_answer = pool.submit(self.chat, messages, image)
                future_answer_result = await future_answer.result()
                model_response = future_answer_result  # 获取第一个选项
                return model_response, ""
        except Exception as e:
            logger.error(f"Error in LlavaNextGenerator.request: {e}")
            logger.info(traceback.format_exc)
            # sys.exit(0)
            return "", str(e)

    def request_sync(self, messages=None, image=None):
        return self.get_response(messages, image)

    async def chat(self, messages, image):
        loop = asyncio.get_event_loop()

        func = partial(self.get_response, messages=messages, image=image)

        return await loop.run_in_executor(None, func)

    def get_response(self, messages, image=None):
        logging.info(f"messages1: {messages}")

        messages_new = []
        messages_new.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": messages[0]["content"]}],
            }
        )
        messages_new.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": messages[-1]["content"]},
                    {"type": "image"},
                ],
            }
        )

        logging.info(f"messages_new: {messages_new}")

        prompt = self.processor.apply_chat_template(
            messages_new, add_generation_prompt=True
        )

        # print('type(image) = ', type(image))

        if image is not None:
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
                "cuda:0"
            )
        else:
            # print('image is None')
            inputs = self.processor(text=prompt, return_tensors="pt").to("cuda:0")

        generation_args = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
        }

        # print(generation_args)

        generate_ids = self.model.generate(**inputs, **generation_args)

        # logger.info(f"generate_ids: {generate_ids}")

        # remove input tokens
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True)[
            0
        ]

        return response


class Qwen2VLGenerator:
    def __init__(
        self, model, processor, max_new_tokens=512, temperature=0.7, do_sample=True
    ):
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

    async def request(self, messages: list = None, image=None):
        try:
            cpu_count = multiprocessing.cpu_count()
            with ThreadPoolExecutor(max_workers=cpu_count * 2) as pool:
                future_answer = pool.submit(self.chat, messages, image)
                future_answer_result = await future_answer.result()
                model_response = future_answer_result  # 获取第一个选项
                return model_response, ""
        except Exception as e:
            logger.error(f"Error in Qwen2VLGenerator.request: {e}")
            logger.info(traceback.format_exc)
            sys.exit(0)
            return "", str(e)

    def request_sync(self, messages=None, image=None):
        return self.get_response(messages, image)

    async def chat(self, messages, image):
        loop = asyncio.get_event_loop()

        func = partial(self.get_response, messages=messages, image=image)

        return await loop.run_in_executor(None, func)

    def get_response(self, messages, image=None):
        logging.info(f"messages1: {messages}")

        system_prompt = messages[0]["content"]
        user_prompt = messages[-1]["content"]

        logging.info(f"user_prompt: {user_prompt}")

        if "<|image_1|>" in user_prompt:
            user_prompt_0, user_prompt_1 = user_prompt.split("<|image_1|>")
        else:
            user_prompt_1 = user_prompt

        messages_new = []
        messages_new.append(
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        )

        if "<|image_1|>" in user_prompt:
            messages_new.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt_0},
                        {"type": "image"},
                        {"type": "text", "text": user_prompt_1},
                    ],
                }
            )
        else:
            messages_new.append(
                {"role": "user", "content": [{"type": "text", "text": user_prompt_1}]}
            )

        logging.info(f"messages_new: {messages_new}")

        prompt = self.processor.apply_chat_template(
            messages_new, add_generation_prompt=True
        )

        if image is not None:
            logging.info("image is not None")
            inputs = self.processor(
                text=prompt, images=[image], padding=True, return_tensors="pt"
            ).to("cuda:0")
        else:
            # print('image is None')
            inputs = self.processor(text=prompt, padding=True, return_tensors="pt").to(
                "cuda:0"
            )

        generation_args = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
        }

        generate_ids = self.model.generate(
            **inputs,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **generation_args,
        )

        # remove input tokens
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        return response


class InternVLGenerator:
    def __init__(
        self, model, tokenizer, max_new_tokens=512, temperature=0.7, do_sample=True
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

    def request_sync(self, messages=None, image=None):
        return self.get_response(messages, image)

    async def chat(self, messages, image):
        loop = asyncio.get_event_loop()

        func = partial(self.get_response, messages=messages, image=image)

        return await loop.run_in_executor(None, func)

    def get_response(self, messages, image=None):
        logging.info(f"messages1: {messages}")

        image = load_image(image)

        system_prompt = messages[0]["content"]
        user_prompt = messages[-1]["content"]

        logging.info(f"user_prompt: {user_prompt}")

        user_prompt_0, user_prompt_1 = user_prompt.split("<|image_1|>")

        generation_args = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
        }

        question = f"{system_prompt} {user_prompt_0} <image> {user_prompt_1}"

        response = self.model.chat(self.tokenizer, image, question, generation_args)

        return response
