"""
Learning Rate Scheduler - Warmup + Cosine Decay

In Baby LLM we used a FIXED learning rate (3e-4).
Real LLMs use a SCHEDULE that changes the learning rate over time!

WHY?
  - Start LOW (warmup): model weights are random, big steps = chaos
  - Go HIGH (peak): model is stable, learn fast!  
  - Slowly DECREASE (decay): fine-tune, don't overshoot

GRADIENT CLIPPING:
  Sometimes gradients get HUGE (exploding gradients).
  We clip them to a max value (like a speed limit on a highway).
  max_norm=1.0 means: if gradient magnitude > 1.0, scale it down.
"""

import math


class CosineScheduler:
    """
    Warmup + Cosine Decay learning rate scheduler.
    Used by GPT-3, LLaMA, and most modern LLMs.
    """

    def __init__(self, optimizer, warmup_steps, total_steps, peak_lr, min_lr=1e-5):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.current_step = 0

    def get_lr(self):
        """Calculate learning rate for current step."""
        step = self.current_step

        if step < self.warmup_steps:
            # WARMUP: linearly increase from 0 to peak_lr
            return self.peak_lr * (step / self.warmup_steps)
        else:
            # COSINE DECAY: smoothly decrease from peak_lr to min_lr
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr + (self.peak_lr - self.min_lr) * cosine

    def step(self):
        """Update the learning rate."""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.current_step += 1
        return lr


def demo_scheduler():
    import torch

    print("=" * 60)
    print("Learning Rate Scheduler Demo")
    print("=" * 60)

    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0)
    scheduler = CosineScheduler(
        optimizer, warmup_steps=200, total_steps=2000,
        peak_lr=3e-4, min_lr=1e-5
    )

    print("\nLR at key points:")
    for target in [0, 50, 100, 200, 500, 1000, 1500, 2000]:
        scheduler.current_step = target
        lr = scheduler.get_lr()
        bar = "#" * int(lr / 3e-4 * 40)
        print(f"  Step {target:5d}: lr={lr:.6f} |{bar}")

    print("\nDone!")


if __name__ == "__main__":
    demo_scheduler()
