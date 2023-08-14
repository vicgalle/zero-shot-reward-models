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

        self.sentiment_fn = pipeline(
            "sentiment-analysis",
            "lvwerra/distilbert-imdb",
            top_k=2,
            truncation=True,
            batch_size=256,
            device=self.device,
            function_to_apply="none",
        )

        self.yes_token_id = 2163  # this is for Flan-T5, change it accordingly
        self.no_token_id = 465  # this is for Flan-T5, change it accordingly

    def reward_fn(self, samples: List[str], **kwargs) -> List[float]:
        scores = []
        for sample in samples:
            score_prompt = []
            for prompt, cl in [
                (
                    f"Review: {sample}\n\nIs this movie review from FilmAffinity? Response:",
                    "yes",
                ),
                (f"Review: {sample}\n\nIs this text too repetitive? Response:", "no"),
            ]:
                x = self.tokenizer([prompt], return_tensors="pt").input_ids.to(
                    self.device
                )
                outputs = self.model.generate(
                    x,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=1,
                )
                v_yes_exp = (
                    torch.exp(outputs.scores[0][:, self.yes_token_id]).cpu().numpy()[0]
                )
                v_no_exp = (
                    torch.exp(outputs.scores[0][:, self.no_token_id]).cpu().numpy()[0]
                )
                if cl == "yes":
                    score = v_yes_exp / (v_yes_exp + v_no_exp)
                else:
                    score = v_no_exp / (v_yes_exp + v_no_exp)
                score_prompt.append(score)

            scores.append(
                (np.mean(score_prompt) - 0.5) * 10
            )  # we do some rescaling to improve PPO.
        return scores

    def metric_fn(self, samples: List[str], **kwargs) -> List[float]:
        """Similar to reward_fn, but without rescaling, to make it interpretable in the logs."""
        scores = []
        scores_positive = []
        for sample in samples:
            score_prompt = []
            for prompt, cl in [
                (
                    f"Review: {sample}\n\nIs this movie review from FilmAffinity? Response:",
                    "yes",
                ),
                (f"Review: {sample}\n\nIs this text too repetitive? Response:", "no"),
            ]:
                x = self.tokenizer([prompt], return_tensors="pt").input_ids.to(
                    self.device
                )
                outputs = self.model.generate(
                    x,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=1,
                )
                v_yes_exp = (
                    torch.exp(outputs.scores[0][:, self.yes_token_id]).cpu().numpy()[0]
                )
                v_no_exp = (
                    torch.exp(outputs.scores[0][:, self.no_token_id]).cpu().numpy()[0]
                )
                if cl == "yes":
                    score = v_yes_exp / (v_yes_exp + v_no_exp)
                else:
                    score = v_no_exp / (v_yes_exp + v_no_exp)
                score_prompt.append(score)

            scores.append(np.mean(score_prompt))
            scores_positive.append(score_prompt[0])
        return {"prob_ensemble": scores, "prob_positive": scores_positive}

    def reward_fn_classifier(self, samples: List[str], **kwargs) -> List[float]:
        sentiments = list(map(get_positive_score, self.sentiment_fn(samples)))
        return sentiments


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


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
