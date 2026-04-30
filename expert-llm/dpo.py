"""
DPO - Direct Preference Optimization
=====================================
RLHF (Level 3) has 3 steps:
  1. Collect human preferences
  2. Train a reward model
  3. Use RL (PPO) to optimize

DPO simplifies this to 1 step:
  1. Collect human preferences
  2. Directly optimize the LLM!
  (No reward model, no RL!)

The DPO loss:
  L = -log(sigmoid(beta * (log_prob_chosen - log_prob_rejected)))

Intuition: Make the model MORE likely to produce
the preferred response and LESS likely to produce
the rejected one. That's it!

Used by: LLaMA-2, Zephyr, many open-source models
"""

import torch
import torch.nn.functional as F


def dpo_loss(model_chosen_logprobs, model_rejected_logprobs,
             ref_chosen_logprobs, ref_rejected_logprobs, beta=0.1):
    """
    Compute DPO loss.
    
    Args:
        model_chosen_logprobs: log P(chosen | model)
        model_rejected_logprobs: log P(rejected | model)
        ref_chosen_logprobs: log P(chosen | reference_model)
        ref_rejected_logprobs: log P(rejected | reference_model)
        beta: temperature parameter
    """
    chosen_rewards = beta * (model_chosen_logprobs - ref_chosen_logprobs)
    rejected_rewards = beta * (model_rejected_logprobs - ref_rejected_logprobs)
    
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    return loss


def demo_dpo():
    print("=" * 60)
    print("DPO - Direct Preference Optimization")
    print("=" * 60)
    print("\n  RLHF: preferences -> reward model -> RL (PPO) -> better LLM")
    print("  DPO:  preferences -> directly optimize LLM (no reward model!)")
    print("\n  DPO is simpler, more stable, and often works just as well!")


if __name__ == "__main__":
    demo_dpo()
