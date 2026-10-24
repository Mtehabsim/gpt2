import torch
from transformer_lens.model_bridge import TransformerBridge


model = TransformerBridge.boot_transformers(
    "openai-community/gpt2",
    device="cpu",
)
model.eval()

print("MODEL")
print(
    f"layers={model.cfg.n_layers}, heads={model.cfg.n_heads}, "
    f"d_model={model.cfg.d_model}, vocab={model.cfg.d_vocab}"
)

prompt = "The capital of France is"
tokens = model.to_tokens(prompt, prepend_bos=False)

print("\nTOKENS")
print("IDs:", tokens.tolist())
print("Pieces:", model.to_str_tokens(tokens))
print("Shape:", tuple(tokens.shape))

with torch.inference_mode():
    logits, cache = model.run_with_cache(tokens)
    print("\nLOGITS")
    print("Shape:", tuple(logits.shape))  # [1, seq_len, vocab_size]

    next_logits = logits[0, -1]  # [vocab_size]
    print("next_logits ", next_logits)
    next_probs = next_logits.softmax(dim=-1)
    print("next_probs ", next_probs)
    values, ids = next_probs.topk(5)
    print("\nTOP 5 NEXT TOKENS")
    for token_id, probability in zip(ids.tolist(), values.tolist()):
        print(
            f"{model.to_single_str_token(token_id)!r:>16}  "
            f"id={token_id:>6}  p={probability:.4%}"
        )
    print("Probability sum:", next_probs.sum().item())

    print("\nACTIVATION CACHE (LAYER 0)")
    for name in ("blocks.0.hook_in", "blocks.0.attn.hook_pattern"):
        print(name, tuple(cache[name].shape))

    print(
        "Head 0, last-position attention:",
        cache["blocks.0.attn.hook_pattern"][0, 0, -1].tolist(),
    )

    generated = tokens.clone()
    max_new_tokens = 50
    eos_token_id = 50256

    for _ in range(max_new_tokens):
        logits = model(generated)
        next_logits = logits[0, -1]

        temperature = 0.8
        top_k = 40
        scaled_logits = next_logits / temperature
        values, indices = scaled_logits.topk(top_k)
        probs = values.softmax(dim=-1)
        sampled_index = torch.multinomial(probs, num_samples=1)
        next_id = indices[sampled_index]

        # probs = next_logits.softmax(dim=-1)
        # next_id = torch.multinomial(probs, num_samples=1)

        if next_id.item() == eos_token_id:
            print("The model outputed the end token, STOPP \n")
            break

        generated = torch.cat([generated, next_id.view(1, 1)], dim=1)

print("\nGENERATED TEXT")
print(repr(model.to_string(generated[0])))
