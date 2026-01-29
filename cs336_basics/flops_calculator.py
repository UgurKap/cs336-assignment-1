def calculate_flops(
    _id: str, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int
):
    # Transformer Block

    # MHA
    qkv_projection = 2 * 3 * context_length * d_model * d_model
    key_query = 2 * context_length * d_model * context_length
    weighted_sum = 2 * context_length * context_length * d_model
    attn_proj = 2 * context_length * d_model * d_model

    mha_flops = qkv_projection + key_query + weighted_sum + attn_proj

    # Feed-Forward
    ff_flops = 6 * context_length * d_model * d_ff  # From the 3 matrix multiplications for W1, W2, W3

    transformer_flops = mha_flops + ff_flops
    total_transformer_flops = num_layers * transformer_flops
    out_proj_flops = 2 * context_length * d_model * vocab_size

    total_flops = total_transformer_flops + out_proj_flops
    print()
    print(f"================================================ {_id} ================================================")
    print()
    print(f"Total FLOPs: {total_flops:.2E}")
    print(
        f"Output Projection FLOPs: {out_proj_flops:.2E} | Relative Contribution to the total FLOPs: {(100 * out_proj_flops / total_flops):.2f}%"
    )
    print(
        f"Transformer Blocks FLOPs: {total_transformer_flops:2E} | Relative Contribution to the total FLOPs: {(100 * total_transformer_flops / total_flops):.2f}%"
    )
    print("\n=====\n")
    print(f"Single Transformer Block FLOPs: {transformer_flops:.2E}")
    print(
        f"Attention FLOPs: {mha_flops:.2E} | Relative Contribution to the transformer block FLOPs: {(100 * mha_flops / transformer_flops):.2f}%"
    )
    print(
        f"Feedforward FLOPs: {ff_flops:.2E} | Relative Contribution to the transformer block FLOPs: {(100 * ff_flops / transformer_flops):.2f}%"
    )
    print()


configs = {
    "GPT-2 Small": {
        "vocab_size": 50_257,
        "context_length": 1024,
        "num_layers": 12,
        "d_model": 768,
        "num_heads": 12,
        "d_ff": 768 * 4,
    },  # Assuming d_ff = 4xd
    "GPT-2 Medium": {
        "vocab_size": 50_257,
        "context_length": 1024,
        "num_layers": 24,
        "d_model": 1024,
        "num_heads": 16,
        "d_ff": 1024 * 4,
    },
    "GPT-2 Large": {
        "vocab_size": 50_257,
        "context_length": 1024,
        "num_layers": 36,
        "d_model": 1280,
        "num_heads": 25,
        "d_ff": 1280 * 4,
    },
    "GPT-2 XL": {
        "vocab_size": 50_257,
        "context_length": 1024,
        "num_layers": 48,
        "d_model": 1600,
        "num_heads": 25,
        "d_ff": 6400,
    },
    "GPT-2 XL Extended Context Length": {
        "vocab_size": 50_257,
        "context_length": 16_384,
        "num_layers": 48,
        "d_model": 1600,
        "num_heads": 25,
        "d_ff": 6400,
    },
}

for k, v in configs.items():
    calculate_flops(_id=k, **v)
