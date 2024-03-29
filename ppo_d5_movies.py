import os
import pathlib
from typing import List

import numpy as np
import torch
import trlx
import yaml
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
from trlx.data.configs import TRLConfig

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
            "ruiqi-zhong/d5_t5_validator_700M"
        ).to(self.device)

        self.yes_token_id = 2163  # this is for Flan-T5, change it accordingly
        self.no_token_id = 465  # this is for Flan-T5, change it accordingly

    def reward_fn(self, samples: List[str], **kwargs) -> List[float]:
        scores = []
        for sample in samples:
            hypothesis = "This review is about a funny sci-fi"

            template = f"""Check whether the TEXT satisfies a PROPERTY. Respond with Yes or No. When uncertain, output No. 

            Now complete the following example -
            input: PROPERTY: {hypothesis}
            TEXT: {sample}
            output:"""

            x = self.tokenizer([template], return_tensors="pt").input_ids.to(
                self.device
            )
            outputs = self.model.generate(
                x, return_dict_in_generate=True, output_scores=True, max_new_tokens=1
            )
            v_yes_exp = (
                torch.exp(outputs.scores[0][:, self.yes_token_id]).cpu().numpy()[0]
            )
            v_no_exp = (
                torch.exp(outputs.scores[0][:, self.no_token_id]).cpu().numpy()[0]
            )
            scores.append(
                (v_yes_exp / (v_yes_exp + v_no_exp) - 0.5) * 10
            )  # we do some rescaling to improve PPO. This is Eq. (3) in the paper
        return scores

    def metric_fn(self, samples: List[str], **kwargs) -> List[float]:
        """Similar to reward_fn, but without rescaling, to make it interpretable in the logs."""
        scores = []
        for sample in samples:
            hypothesis = "This review is about a funny sci-fi"

            template = f"""Check whether the TEXT satisfies a PROPERTY. Respond with Yes or No. When uncertain, output No. 

            Now complete the following example -
            input: PROPERTY: {hypothesis}
            TEXT: {sample}
            output:"""

            x = self.tokenizer([template], return_tensors="pt").input_ids.to(
                self.device
            )
            outputs = self.model.generate(
                x, return_dict_in_generate=True, output_scores=True, max_new_tokens=1
            )
            v_yes_exp = (
                torch.exp(outputs.scores[0][:, self.yes_token_id]).cpu().numpy()[0]
            )
            v_no_exp = (
                torch.exp(outputs.scores[0][:, self.no_token_id]).cpu().numpy()[0]
            )
            scores.append(v_yes_exp / (v_yes_exp + v_no_exp))
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
