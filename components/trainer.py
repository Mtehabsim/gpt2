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
        self.sampler = solutions.TransformerSampler(
            self.model, reference_gpt2.tokenizer
        )
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
        self, batch: dict[str, Int[Tensor, "batch seq"]]
    ) -> Float[Tensor, ""]:
        """
        Calculates the loss on the tokens in the batch, performs a gradient update step, and logs the loss.

        Remember that `batch` is a dictionary with the single key 'tokens'.
        """
        raise NotImplementedError()
        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        """
        Evaluate the model on the test set and return the accuracy.
        """
        self.model.eval()
        #
        # YOUR CODE HERE - fill in the `evaluate` method
        #
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

        print(self.sampler.sample(self.args.eval_prompt, max_tokens_generated=50))
        for epoch in range(self.args.epochs):
            for i, batch in enumerate(self.train_loader):
                loss = self.training_step(batch)
                progress_bar.update()
                progress_bar.set_description(
                    f"Epoch {epoch + 1}, loss: {loss:.3f}, accuracy: {accuracy:.3f}"
                )
                if i >= self.args.max_steps_per_epoch:
                    break

            accuracy = self.evaluate()
            print(self.sampler.sample(self.args.eval_prompt, max_tokens_generated=50))

        wandb.finish()
