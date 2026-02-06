from cs336_basics import decoder, training_utils, transformer_modules
from cs336_basics.bpe_tokenizer import Tokenizer
import json
import torch
import typer

def main(
    model_id: str,
    vocab_path: str = "/home/ugurkap/stanford-cs336-assignments/assignment1-basics/tiny_vocab.pickle",
    merges_path: str = "/home/ugurkap/stanford-cs336-assignments/assignment1-basics/tiny_merges.pickle",
    start_prompt: str = "Once upon a time",
    max_new_tokens: int = 1024,
    temperature: float = 0.3,
    top_p: float = 0.8,
):
    with open(f"/home/ugurkap/stanford-cs336-assignments/assignment1-basics/models/{model_id}/config.json") as f:
        config = json.load(f)

    model = transformer_modules.TransformerLM(**config["model"]["params"])
    model.to("cuda")
    model = torch.compile(model)
    opt = training_utils.AdamW(model.parameters(), **config["optimizer"]["params"])

    training_utils.load_checkpoint(
        f"/home/ugurkap/stanford-cs336-assignments/assignment1-basics/models/{model_id}/standard-rope_checkpoint_final.pt",
        model,
        opt,
    )

    tok = Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])

    print(
        decoder.generate_text(
            start_prompt, model, tok, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p
        )
    )

if __name__ == "__main__":
    typer.run(main)
