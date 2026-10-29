import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import datasets
import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import wandb
from IPython.display import display
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.utilities import gelu_new, tokenize_and_concatenate
from transformers import PreTrainedTokenizerFast

from components import Config, DemoTransformer

device = t.device(
    "mps"
    if t.backends.mps.is_available()
    else "cuda"
    if t.cuda.is_available()
    else "cpu"
)


reference_gpt2 = TransformerBridge.boot_transformers(
    "gpt2",
    device=device,
)
reference_gpt2.eval()
model_cfg = Config(
    debug=False,
    d_model=32,
    n_heads=16,
    d_head=2,
    d_mlp=32 * 4,
    n_layers=4,
    n_ctx=128,
    d_vocab=reference_gpt2.cfg.d_vocab,
)
model = DemoTransformer(model_cfg).to(device)


@dataclass
class TransformerTrainingArgs:
    batch_size: int = 32
    epochs: int = 10
    max_steps_per_epoch: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-2
    wandb_project: str | None = "day1-demotransformer"
    wandb_name: str | None = None
    eval_prompt: str = "Once upon a time"


args = TransformerTrainingArgs()

dataset = datasets.load_dataset("roneneldan/TinyStories", split="train")
# Keep the first 200k of the 2.1M stories, as we only need a subset for training.
dataset = dataset.select(range(200_000))
print(dataset)
print(dataset[0]["text"])

tokenized_dataset = tokenize_and_concatenate(
    dataset,
    reference_gpt2.tokenizer,
    streaming=False,
    max_length=model.cfg.n_ctx,
    column_name="text",
    add_bos_token=True,
    num_proc=8,
)

dataset_dict = tokenized_dataset.train_test_split(test_size=1000)
train_loader = DataLoader(
    dataset_dict["train"],
    batch_size=args.batch_size,
    shuffle=True,
    # num_workers=4, runs faster, but kills the kernel if interrupted
    # pin_memory=False
)
test_loader = DataLoader(
    dataset_dict["test"],
    batch_size=args.batch_size,
    shuffle=False,
    # num_workers=4,
    # pin_memory=False
)

first_batch = train_loader.dataset[: args.batch_size]

print(first_batch.keys())
print(first_batch["tokens"].shape)


import torch as t
import numpy as np
import wandb

from torch import Tensor
from torch.utils.data import DataLoader
from jaxtyping import Float, Int
from tqdm.auto import tqdm
from dataclasses import dataclass

from transformer_lens.model_bridge import TransformerBridge

from components import Config, DemoTransformer


class TransformerTrainer:
    def __init__(self, args: TransformerTrainingArgs, model: DemoTransformer):
        super().__init__()
        self.model = model
        self.args = args
        self.optimizer = t.optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        self.step = 0

        self.train_loader = DataLoader(
            dataset_dict["train"],
            batch_size=args.batch_size,
            shuffle=True,
            # num_workers=4, runs faster, but kills the kernel if interrupted
            # pin_memory=False,
        )
        self.test_loader = DataLoader(
            dataset_dict["test"],
            batch_size=args.batch_size,
            shuffle=False,
            # num_workers=4,
            # pin_memory=False,
        )

    def training_step(
        self,
        batch: dict[str, Int[Tensor, "batch seq"]],  # type: ignore
    ) -> Float[Tensor, ""]:  # type: ignore
        """
        Calculates the loss on the tokens in the batch, performs a gradient update step, and logs the loss.

        Remember that `batch` is a dictionary with the single key 'tokens'.
        """

        self.model.train()
        tokens = batch["tokens"]
        tokens = tokens.to(next(self.model.parameters()).device)

        # forward pass the model
        logits = self.model(tokens)
        prediction_logits = logits[:, :-1, :]
        targets = tokens[:, 1:]
        flat_logits = prediction_logits.reshape(-1, self.model.cfg.d_vocab)
        flat_targets = targets.reshape(-1)
        loss = F.cross_entropy(flat_logits, flat_targets)
        self.optimizer.zero_grad()
        loss.backward()
        # calculate gradient
        self.optimizer.step()
        self.step += 1
        wandb.log(
            {
                "train_loss": loss.item(),
                "learning_rate": self.args.lr,
            },
            step=self.step,
        )

        return loss.detach()

    @t.inference_mode()
    def evaluate(self) -> float:
        self.model.eval()

        total_correct = 0
        total_predictions = 0

        for batch in self.test_loader:
            tokens = batch["tokens"].to(next(self.model.parameters()).device)

            logits = self.model(tokens)

            prediction_logits = logits[:, :-1, :]
            targets = tokens[:, 1:]

            predictions = prediction_logits.argmax(dim=-1)

            correct = predictions == targets
            total_correct += correct.sum().item()
            total_predictions += targets.numel()

        accuracy = total_correct / total_predictions

        self.model.train()
        return accuracy

    def train(self):
        """
        Trains the model, for `self.args.epochs` epochs. Also handles wandb initialisation, and early stopping
        for each epoch at `self.args.max_steps_per_epoch` steps.
        """
        wandb.init(
            project=self.args.wandb_project, name=self.args.wandb_name, config=self.args
        )
        accuracy = np.nan

        progress_bar = tqdm(total=self.args.max_steps_per_epoch * self.args.epochs)

        for epoch in range(self.args.epochs):
            for i, batch in enumerate(self.train_loader):
                loss = self.training_step(batch)
                progress_bar.update()
                progress_bar.set_description(
                    f"Epoch {epoch + 1}, loss: {loss:.3f}, accuracy: {accuracy:.3f}"
                )
                if i + 1 >= self.args.max_steps_per_epoch:
                    break

            accuracy = self.evaluate()

            wandb.log(
                {"validation_accuracy": accuracy},
                step=self.step,
            )

        wandb.finish()


args = TransformerTrainingArgs(
    batch_size=50,
    epochs=10,
    max_steps_per_epoch=10,
)
trainer = TransformerTrainer(args, model)
trainer.train()
