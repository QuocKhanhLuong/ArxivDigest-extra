import dataclasses
import logging
import math
import os
import io
import sys
import time
import json
from typing import Optional, Sequence, Union, Dict, Any

import openai
import tqdm
import copy

# Handle both old and new OpenAI SDK versions
try:
    from openai import openai_object
    StrOrOpenAIObject = Union[str, openai_object.OpenAIObject]
    OPENAI_OLD_API = True
except ImportError:
    StrOrOpenAIObject = Union[str, Dict[str, Any]]
    OPENAI_OLD_API = False


openai_org = os.getenv("OPENAI_ORG")
if openai_org is not None:
    openai.organization = openai_org
    logging.warning(f"Switching to organization: {openai_org} for OAI API key.")


@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    #max_tokens: int = 1800
    max_tokens: int = 5400
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    # logprobs: Optional[int] = None


def openai_completion(
    prompts, #: Union[str, Sequence[str], Sequence[dict[str, str]], dict[str, str]],
    decoding_args: OpenAIDecodingArguments,
    model_name="text-davinci-003",
    sleep_time=15,
    batch_size=1,
    max_instances=sys.maxsize,
    max_batches=sys.maxsize,
    return_text=False,
    **decoding_kwargs,
) -> Union[Union[StrOrOpenAIObject], Sequence[StrOrOpenAIObject], Sequence[Sequence[StrOrOpenAIObject]],]:
    """Decode with OpenAI API.

    Args:
        prompts: A string or a list of strings to complete. If it is a chat model the strings should be formatted
            as explained here: https://github.com/openai/openai-python/blob/main/chatml.md. If it is a chat model
            it can also be a dictionary (or list thereof) as explained here:
            https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
        decoding_args: Decoding arguments.
        model_name: Model name. Can be either in the format of "org/model" or just "model".
        sleep_time: Time to sleep once the rate-limit is hit.
        batch_size: Number of prompts to send in a single request. Only for non chat model.
        max_instances: Maximum number of prompts to decode.
        max_batches: Maximum number of batches to decode. This argument will be deprecated in the future.
        return_text: If True, return text instead of full completion object (which contains things like logprob).
        decoding_kwargs: Additional decoding arguments. Pass in `best_of` and `logit_bias` if you need them.

    Returns:
        A completion or a list of completions.
        Depending on return_text, return_openai_object, and decoding_args.n, the completion type can be one of
            - a string (if return_text is True)
            - an openai_object.OpenAIObject object (if return_text is False)
            - a list of objects of the above types (if decoding_args.n > 1)
    """
    is_chat_model = "gpt-3.5" in model_name or "gpt-4" in model_name
    is_single_prompt = isinstance(prompts, (str, dict))
    if is_single_prompt:
        prompts = [prompts]

    if max_batches < sys.maxsize:
        logging.warning(
            "`max_batches` will be deprecated in the future, please use `max_instances` instead."
            "Setting `max_instances` to `max_batches * batch_size` for now."
        )
        max_instances = max_batches * batch_size

    prompts = prompts[:max_instances]
    num_prompts = len(prompts)
    prompt_batches = [
        prompts[batch_id * batch_size : (batch_id + 1) * batch_size]
        for batch_id in range(int(math.ceil(num_prompts / batch_size)))
    ]

    completions = []
    for batch_id, prompt_batch in tqdm.tqdm(
        enumerate(prompt_batches),
        desc="prompt_batches",
        total=len(prompt_batches),
    ):
        batch_decoding_args = copy.deepcopy(decoding_args)  # cloning the decoding_args

        backoff = 5

        while True:
            try:
                time.sleep(3)
                shared_kwargs = dict(
                    model=model_name,
                    **batch_decoding_args.__dict__,
                    **decoding_kwargs,
                )
                
                if OPENAI_OLD_API:
                    # Use old API format
                    if is_chat_model:
                        completion_batch = openai.ChatCompletion.create(
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": prompt_batch[0]}
                            ],
                            **shared_kwargs
                        )
                    else:
                        completion_batch = openai.Completion.create(prompt=prompt_batch, **shared_kwargs)
                    
                    choices = completion_batch.choices
                    
                    for choice in choices:
                        choice["total_tokens"] = completion_batch.usage.total_tokens
                else:
                    # Use new API format
                    client = openai.OpenAI()
                    
                    if is_chat_model:
                        completion_batch = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": prompt_batch[0]}
                            ],
                            temperature=batch_decoding_args.temperature,
                            max_tokens=batch_decoding_args.max_tokens,
                            top_p=batch_decoding_args.top_p,
                            n=batch_decoding_args.n,
                            stream=batch_decoding_args.stream,
                            presence_penalty=batch_decoding_args.presence_penalty,
                            frequency_penalty=batch_decoding_args.frequency_penalty,
                            **decoding_kwargs
                        )
                        
                        # Convert completion to dictionary format for consistency
                        choices = []
                        for choice in completion_batch.choices:
                            choice_dict = {
                                "message": {
                                    "content": choice.message.content,
                                    "role": choice.message.role
                                },
                                "index": choice.index,
                                "finish_reason": choice.finish_reason,
                                "total_tokens": completion_batch.usage.total_tokens
                            }
                            choices.append(choice_dict)
                    else:
                        completion_batch = client.completions.create(
                            model=model_name,
                            prompt=prompt_batch, 
                            temperature=batch_decoding_args.temperature,
                            max_tokens=batch_decoding_args.max_tokens,
                            top_p=batch_decoding_args.top_p,
                            n=batch_decoding_args.n,
                            stream=batch_decoding_args.stream,
                            presence_penalty=batch_decoding_args.presence_penalty,
                            frequency_penalty=batch_decoding_args.frequency_penalty,
                            **decoding_kwargs
                        )
                        
                        # Convert completion to dictionary format for consistency
                        choices = []
                        for choice in completion_batch.choices:
                            choice_dict = {
                                "text": choice.text,
                                "index": choice.index,
                                "finish_reason": choice.finish_reason,
                                "total_tokens": completion_batch.usage.total_tokens
                            }
                            choices.append(choice_dict)
                
                completions.extend(choices)
                break
            except Exception as e:
                logging.warning(f"OpenAI API Error: {e}.")
                if "Please reduce your prompt" in str(e):
                    batch_decoding_args.max_tokens = int(batch_decoding_args.max_tokens * 0.8)
                    logging.warning(f"Reducing target length to {batch_decoding_args.max_tokens}, Retrying...")
                elif not backoff:
                    logging.error("Hit too many failures, exiting")
                    raise e
                else:
                    backoff -= 1
                    logging.warning("Hit request rate limit; retrying...")
                    time.sleep(sleep_time)  # Annoying rate limit on requests.
                    continue

    if return_text:
        if is_chat_model:
            completions = [completion.get("message", {}).get("content", "") for completion in completions]
        else:
            completions = [completion.get("text", "") for completion in completions]
            
    if decoding_args.n > 1:
        # make completions a nested list, where each entry is a consecutive decoding_args.n of original entries.
        completions = [completions[i : i + decoding_args.n] for i in range(0, len(completions), decoding_args.n)]
    if is_single_prompt:
        # Return non-tuple if only 1 input and 1 generation.
        (completions,) = completions
    return completions


def write_ans_to_file(ans_data, file_prefix, output_dir="./output"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, file_prefix + ".txt")
    with open(filename, "w") as f:
        for ans in ans_data:
            f.write(ans + "\n")
