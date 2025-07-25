import os
import sys
import openai
import asyncio
from functools import partial
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from sanic.log import logger
from evals.mind2web_live_eval.agent.Utils import *
from .token_cal import truncate_messages_based_on_estimated_tokens
from openai import AzureOpenAI
import traceback


class GPTGenerator:
    def __init__(self, args, model=None):
        self.model = model
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def request(
        self, messages: list = None, max_tokens: int = 500, temperature: float = 0.7
    ):
        try:
            if "gpt-3.5" in self.model:
                messages = truncate_messages_based_on_estimated_tokens(
                    messages, max_tokens=16385
                )
            cpu_count = multiprocessing.cpu_count()
            with ThreadPoolExecutor(max_workers=cpu_count * 2) as pool:
                future_answer = pool.submit(
                    self.chat, messages, max_tokens, temperature
                )
                future_answer_result = await future_answer.result()
                choice = future_answer_result.choices[0]
                if choice.finish_reason == "length":
                    logger.warning(
                        "Response may be truncated due to length. Be cautious when parsing JSON."
                    )
                openai_response = choice.message.content
                return openai_response, ""
        except Exception as e:
            logger.error(f"Error in GPTGenerator.request: {e}")
            logger.info(traceback.format_exc)
            sys.exit(0)
            return "", str(e)

    def request_sync(
        self,
        messages: list = None,
        image=None,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ):
        data = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if hasattr(self, "response_format"):
            data["response_format"] = self.response_format

        # logger.info('GPT payload: {}'.format(data))

        num_tries = 0

        for i in range(3):
            try:
                completion = self.client.chat.completions.create(**data)
                a = json.loads(completion.json())
                response = a["choices"][0]["message"]["content"]
            except:
                response = ""
                num_tries += 1

        if num_tries == 3:
            logging.info("Error in GPTGenerator.request_sync")

        return response

    async def chat(self, messages, max_tokens=500, temperature=0.7):
        loop = asyncio.get_event_loop()
        data = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if hasattr(self, "response_format"):
            data["response_format"] = self.response_format

        print("data: ", data)

        func = partial(self.client.chat.completions.create, **data)
        return await loop.run_in_executor(None, func)


class JSONModeMixin(GPTGenerator):
    """
    A mixin to add JSON mode support to GPTGenerator classes.
    """

    def __init__(self, model=None):
        super().__init__(model=model)  # Ensure initialization from base class
        self.response_format = {
            "type": "json_object"
        }  # Set response format to JSON object

    @staticmethod
    def prepare_messages_for_json_mode(messages):
        # Ensure there's a system message instructing the model to generate JSON
        if not any(
            "json" in message.get("content", "").lower() for message in messages
        ):
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": "You are a helpful assistant designed to output json.",
                },
            )
        return messages

    async def request(
        self, messages: list = None, max_tokens: int = 500, temperature: float = 0.7
    ):
        messages = self.prepare_messages_for_json_mode(
            messages
        )  # Prepare messages for JSON mode
        return await super().request(messages, max_tokens, temperature)


class GPTGeneratorWithJSON(JSONModeMixin):
    def __init__(self, model=None):
        super().__init__(model=model if model is not None else "gpt-4-turbo")
