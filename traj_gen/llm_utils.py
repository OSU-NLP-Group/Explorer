import traceback
import requests
import time
import logging
import os


class CredentialException(Exception):
    pass


def call_gpt4v(args, messages, max_tokens=2048, temperature=0.01):
    api_key = os.getenv("OPENAI_API_KEY")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    max_num_trial = 3
    num_trial = 0
    call_api_success = True

    while num_trial < max_num_trial:
        try:
            url = "https://api.openai.com/v1/chat/completions"

            payload = {
                "model": args.deployment,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            a = requests.post(url, headers=headers, json=payload).json()
            ans_1st_pass = a["choices"][0]["message"]["content"]
            break
        except:
            logging.info("retry call gptv {}".format(num_trial))
            logging.info(a)
            logging.info(traceback.format_exc())
            num_trial += 1
            ans_1st_pass = ""
            time.sleep(10)

    if num_trial == max_num_trial:
        call_api_success = False

    return ans_1st_pass, call_api_success
