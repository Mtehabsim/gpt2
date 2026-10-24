import torch as t
from transformer_lens.model_bridge import TransformerBridge

from components import Config, DemoTransformer


def test_pretrained_gpt2():
    reference = TransformerBridge.boot_transformers("gpt2", device="cpu")
    reference.eval()

    model = DemoTransformer(Config(debug=False))
    model.eval()

    # Inspect the pretrained parameters
    pretrained = reference.tl_parameters()
    model_params = dict(model.named_parameters())

    missing = sorted(set(model_params) - set(pretrained))
    print("Parameters without matching pretrained keys:", missing)

    # Don't compare randomly initialized models!
    # Next: copy matching pretrained weights into your model,
    # checking shapes and missing keys before comparing outputs.
    assert not missing

    # 1. Verify shapes
    for name, param in model_params.items():
        assert param.shape == pretrained[name].shape, (
            name,
            param.shape,
            pretrained[name].shape,
        )

    # 2. Copy pretrained weights
    with t.no_grad():
        for name, param in model_params.items():
            param.copy_(pretrained[name])

    # 3. Compare predictions
    tokens = reference.to_tokens(
        "The capital of France is",
        prepend_bos=False,
    )

    with t.inference_mode():
        reference_logits, _ = reference.run_with_cache(tokens)
        your_logits = model(tokens)

    difference = (reference_logits - your_logits).abs()

    print("Maximum difference:", difference.max().item())
    print("Mean difference:", difference.mean().item())

    assert t.allclose(reference_logits, your_logits, atol=1e-4, rtol=1e-4)
