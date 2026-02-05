from .transformer_modules import softmax
from .bpe_tokenizer import Tokenizer
import torch
from torch.nn import Module


def generate_text(
    user_prompt: str,
    model: Module,
    tokenizer: Tokenizer,
    ctx_length: int = 256,
    eot_token: str = "<|endoftext|>",
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eps: float = 1e-8,
):
    token_ids = torch.tensor(tokenizer.encode(user_prompt), dtype=torch.long, device="cuda")
    eot_id = tokenizer.encode(eot_token)[0]
    for _ in range(max_new_tokens):
        ctx_ids = token_ids[-ctx_length:]
        with torch.no_grad():
            model_out = model(ctx_ids.view(1, -1))
        probs = softmax(model_out[0, -1, :] / (temperature + eps), -1)
        sorted_probs, sorted_indices = probs.topk(len(tokenizer))
        sorted_probs_sum = torch.cumsum(sorted_probs, -1)
        remove_mask = sorted_probs_sum > top_p
        if remove_mask.all() == True:
            remove_mask[0] = False
        model_out[0, -1, :][sorted_indices[remove_mask]] = -torch.inf
        probs = softmax(model_out[0, -1, :], -1)
        next_token = torch.multinomial(probs, 1)
        if next_token == eot_id:
            break
        token_ids = torch.cat((token_ids, next_token))
    return tokenizer.decode(token_ids.tolist())