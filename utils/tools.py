import tiktoken, json, re, textwrap, yaml
from typing import List, Dict, Iterable
from langchain_openai import ChatOpenAI

MAX_TOKENS = 16000
SPLIT_TOKENS = MAX_TOKENS * 0.5

def length_of_tokens(prompt, encoding_name: str = "cl100k_base"):
    enc = tiktoken.get_encoding(encoding_name)
    all_tokens = enc.encode(prompt, disallowed_special=())
    return len(all_tokens)


def batch_chunks(chunks: List[str], prompt:str = "", params_name:str = "chunk_list_content", extra_vars: Dict[str, str] | None = None) -> Iterable[List[str]]:
    """
    将 chunks 切分成若干批，使得 `prompt + batch` 的 token
    不超过 BATCH_TARGET。
    """
    batch: List[str] = []
    cur_tokens = 0
    extra_vars = extra_vars or {}
    param_format= {**extra_vars, params_name: ""}
    prompt_tokens = length_of_tokens(
        prompt.format(**param_format)
    )

    for chunk in chunks:
        chunk_tokens = length_of_tokens(chunk)
        if (
            cur_tokens + chunk_tokens + prompt_tokens > SPLIT_TOKENS
            and batch
        ):
            yield batch
            batch = []
            cur_tokens = 0
        batch.append(chunk)
        cur_tokens += chunk_tokens
    if batch:
        yield batch


def split_prompt(
    prompt: str,
    max_tokens: int = SPLIT_TOKENS,
    encoding_name: str = "cl100k_base"
) -> List[str]:
    max_tokens = int(max_tokens)
    enc = tiktoken.get_encoding(encoding_name)
    all_tokens = enc.encode(prompt, disallowed_special=())
    chunks: List[str] = []
    
    for i in range(0, len(all_tokens), max_tokens):
        chunk_tokens = all_tokens[i : i + max_tokens]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)

    return chunks


