# Zero-Shot Reward Models

This repository showcases a generic technique to use an instruction-tuned LLM such as FLAN-T5 as a reward model for RLHF tasks. It relies in the trlx library

## Explanation

We can use an instruction-tuned LLM, such as FLAN-T5, as a reward model by framing the prompt as a yes/no question. Then, we obtain the logits corresponding to the tokens for "yes" and "no" and normalize them to obtain the probability of the answer being "yes". This probability is then used as the reward:

```python
class ZeroShotRewardModel:
    
    def reward_fn(self, samples: List[str], **kwargs) -> List[float]:
        scores = []
        for sample in samples:
            input_text = f"Review: {sample}\n\n Is this movie review positive? Response:"
            x = self.tokenizer([input_text], return_tensors="pt").input_ids.to(self.device)
            outputs = self.model.generate(x, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
            p_yes = torch.exp(outputs.scores[0][:, 2163]).cpu().numpy()[0]
            p_no = torch.exp(outputs.scores[0][:, 465]).cpu().numpy()[0]
            scores.append(p_yes / (p_yes + p_no))
        return scores

```

## Example: optimizing for positive sentiment


```
python ppo_flan_sentiments.py
```

At the beginning, the model to be optimized has these metrics:

```
                                                        Evaluation #0 reward/mean: 0.0588 metrics/prob_positive: 0.506                                                         
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ prompt                         ┃ output                                                                                                             ┃ reward ┃ prob_positive ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Great artists, always suffered │ ).Most of this film arouses your interest only by accusing of!!ESP, the articles have gone on about Orson Welles   │ -3.34  │ 0.166         │
│                                │ during the reigns of his Niven film, by other '                                                                    │        │               │
├────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┼───────────────┤
│ What do you get                │  out of it all? Room context? Maya Fey plays a type of girl she sees brought back to life in spades. Why would     │ -4.87  │ 0.0129        │
│                                │ anyone attempt to do something as insipid as "The Wicked                                                           │        │               │
├────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┼───────────────┤
│ Should we take the             │  ages for dialogue when this game turns out to be so bad? Did Jim Simons really think this could make it? One of   │ -3.05  │ 0.195         │
│                                │ the reasons I personally admire it is that it's dark and often weird                                               │        │               │
└────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴────────┴───────────────┘
```