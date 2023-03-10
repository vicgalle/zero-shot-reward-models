import os
import pathlib
from typing import List

import torch
import yaml
from datasets import load_dataset
from transformers import pipeline

import trlx
from trlx.data.configs import TRLConfig

import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer


config_path = pathlib.Path(__file__).parent.joinpath(
    "./configs/ppo_flan_sentiments.yml"
)
with config_path.open() as f:
    default_config = yaml.safe_load(f)


class ZeroShotRewardModel:
    def __init__(self) -> None:
        if torch.cuda.is_available():
            self.device = int(os.environ.get("LOCAL_RANK", 0))

        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        self.model = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-large"
        ).to(self.device)

    def reward_fn(self, samples: List[str], **kwargs) -> List[float]:
        scores = []
        for sample in samples:
            input_text = (
                f"Review: {sample}\n\n Is this movie review positive? Response:"
            )
            x = self.tokenizer([input_text], return_tensors="pt").input_ids.to(
                self.device
            )
            outputs = self.model.generate(
                x, return_dict_in_generate=True, output_scores=True, max_new_tokens=1
            )
            p_yes = torch.exp(outputs.scores[0][:, 2163]).cpu().numpy()[0]
            p_no = torch.exp(outputs.scores[0][:, 465]).cpu().numpy()[0]
            scores.append(
                (p_yes / (p_yes + p_no) - 0.5) * 10
            )  # we do some rescaling to improve PPO.
        return scores

    def metric_fn(self, samples: List[str], **kwargs) -> List[float]:
        """Similar to reward_fn, but without rescaling, to make it interpretable in the logs."""
        scores = []
        for sample in samples:
            input_text = (
                f"Review: {sample}\n\n Is this movie review positive? Response:"
            )
            x = self.tokenizer([input_text], return_tensors="pt").input_ids.to(
                self.device
            )
            outputs = self.model.generate(
                x, return_dict_in_generate=True, output_scores=True, max_new_tokens=1
            )
            p_yes = torch.exp(outputs.scores[0][:, 2163]).cpu().numpy()[0]
            p_no = torch.exp(outputs.scores[0][:, 465]).cpu().numpy()[0]
            scores.append(p_yes / (p_yes + p_no))
        return {"prob_positive": scores}


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    # Load the reward model
    reward_model = ZeroShotRewardModel()

    # Take few words off of movies reviews as prompts
    imdb = load_dataset("imdb", split="train+test")

    trlx.train(
        reward_fn=reward_model.reward_fn,
        prompts=[" ".join(review.split()[:4]) for review in imdb["text"][:-64]],
        metric_fn=reward_model.metric_fn,
        eval_prompts=[" ".join(review.split()[:4]) for review in imdb["text"][-64:]],
        config=config,
    )


if __name__ == "__main__":
    main()
