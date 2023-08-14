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


config_path = pathlib.Path(__file__).parent.joinpath("./configs/ppo_flan_detox.yml")
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
            "facebook/roberta-hate-speech-dynabench-r4-target",
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
            input_text = (
                f"Text: {sample}\n\n Does this text contain toxic speech? Response:"
            )
            x = self.tokenizer([input_text], return_tensors="pt").input_ids.to(
                self.device
            )
            outputs = self.model.generate(
                x, return_dict_in_generate=True, output_scores=True, max_new_tokens=1
            )
            p_yes = torch.exp(outputs.scores[0][:, self.yes_token_id]).cpu().numpy()[0]
            p_no = torch.exp(outputs.scores[0][:, self.no_token_id]).cpu().numpy()[0]
            scores.append(
                (p_no / (p_yes + p_no) - 0.5) * 10
            )  # we do some rescaling to improve PPO.
        return scores

    def metric_fn(self, samples: List[str], **kwargs) -> List[float]:
        """Similar to reward_fn, but without rescaling, to make it interpretable in the logs."""
        scores = []
        for sample in samples:
            input_text = (
                f"Text: {sample}\n\n Does this text contain toxic speech? Response:"
            )
            x = self.tokenizer([input_text], return_tensors="pt").input_ids.to(
                self.device
            )
            outputs = self.model.generate(
                x, return_dict_in_generate=True, output_scores=True, max_new_tokens=1
            )
            p_yes = torch.exp(outputs.scores[0][:, self.yes_token_id]).cpu().numpy()[0]
            p_no = torch.exp(outputs.scores[0][:, self.no_token_id]).cpu().numpy()[0]
            scores.append(p_no / (p_yes + p_no))
        return {"prob_positive": scores}

    def reward_fn_classifier(self, samples: List[str], **kwargs) -> List[float]:
        sentiments = list(map(get_positive_score, self.sentiment_fn(samples)))
        return sentiments


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["nothate"]


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    # Load the reward model
    reward_model = ZeroShotRewardModel()

    # Take few words off of movies reviews as prompts
    # imdb = load_dataset("imdb", split="train+test")
    ds = load_dataset("allenai/real-toxicity-prompts", split="train")

    def filter_fn(sample):
        toxicity = sample["prompt"]["toxicity"]
        return toxicity is not None and toxicity > 0.3

    ds = ds.filter(filter_fn, batched=False)

    trlx.train(
        reward_fn=reward_model.reward_fn_classifier,
        prompts=[data["text"] for data in ds["prompt"][:-64]],
        metric_fn=reward_model.metric_fn,
        eval_prompts=[data["text"] for data in ds["prompt"][-64:]],
        config=config,
    )


if __name__ == "__main__":
    main()
